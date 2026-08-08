import os
from typing import Dict

import pandas as pd
import psycopg2
from scipy.stats import norm


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "assistant_analytics"),
        user=os.getenv("POSTGRES_USER", "analyst"),
        password=os.getenv("POSTGRES_PASSWORD", "analyst_pw"),
    )


def build_segment_funnel(conn) -> pd.DataFrame:
    """Build one segmented funnel aggregation by onboarding variant.

    Metrics:
      (a) session-level first-query prompt-failure rate:
            prompt failure == response_status = 'no_prompt_entered'
            first query per session is the earliest query_timestamp
            denominator = sessions that have at least one query

      (b) answered-query rate:
            answered queries / total query rows

      (c) share of sessions with at least one answered query:
            sessions with any response_status='answered' / sessions that have at least one query

    Returns a DataFrame with one row per onboarding_variant.
    """

    sql = """
    WITH session_queries AS (
      SELECT
        s.session_id,
        uo.onboarding_variant,
        q.query_id,
        q.query_timestamp,
        q.response_status
      FROM assistant_sessions s
      JOIN user_onboarding uo
        ON uo.user_id = s.user_id
      JOIN assistant_queries q
        ON q.session_id = s.session_id
    ),

    first_query_per_session AS (
      SELECT DISTINCT ON (sq.session_id)
        sq.session_id,
        sq.onboarding_variant,
        sq.response_status AS first_response_status
      FROM session_queries sq
      ORDER BY sq.session_id, sq.query_timestamp ASC, sq.query_id ASC
    ),

    variant_session_prompt AS (
      SELECT
        onboarding_variant,
        COUNT(DISTINCT session_id) AS sessions_with_queries,
        SUM(CASE WHEN first_response_status = 'no_prompt_entered' THEN 1 ELSE 0 END)
          AS sessions_first_query_prompt_failures
      FROM first_query_per_session
      GROUP BY onboarding_variant
    ),

    variant_query_rates AS (
      SELECT
        onboarding_variant,
        COUNT(*) AS total_queries,
        SUM(CASE WHEN response_status = 'answered' THEN 1 ELSE 0 END) AS answered_queries,
        SUM(CASE WHEN response_status = 'answered' THEN 1 ELSE 0 END)::numeric
          / NULLIF(COUNT(*), 0) AS answered_query_rate
      FROM session_queries
      GROUP BY onboarding_variant
    ),

    session_answered AS (
      SELECT
        session_id,
        onboarding_variant,
        BOOL_OR(response_status = 'answered') AS has_answered
      FROM session_queries
      GROUP BY session_id, onboarding_variant
    ),

    variant_session_answered AS (
      SELECT
        onboarding_variant,
        COUNT(DISTINCT session_id) AS sessions_with_queries,
        SUM(CASE WHEN has_answered THEN 1 ELSE 0 END) AS sessions_with_at_least_one_answered,
        SUM(CASE WHEN has_answered THEN 1 ELSE 0 END)::numeric
          / NULLIF(COUNT(DISTINCT session_id), 0) AS share_sessions_with_answered
      FROM session_answered
      GROUP BY onboarding_variant
    )

    SELECT
      vsp.onboarding_variant,
      vsp.sessions_with_queries,
      vsp.sessions_first_query_prompt_failures,
      vsp.sessions_first_query_prompt_failures::numeric
        / NULLIF(vsp.sessions_with_queries, 0) AS session_prompt_failure_rate,
      vqr.total_queries,
      vqr.answered_queries,
      vqr.answered_query_rate,
      vsa.sessions_with_at_least_one_answered,
      vsa.share_sessions_with_answered
    FROM variant_session_prompt vsp
    JOIN variant_query_rates vqr
      ON vqr.onboarding_variant = vsp.onboarding_variant
    JOIN variant_session_answered vsa
      ON vsa.onboarding_variant = vsp.onboarding_variant
    ORDER BY vsp.onboarding_variant;
    """

    return pd.read_sql_query(sql, conn)


def run_significance_test(funnel_df: pd.DataFrame) -> Dict[str, float]:
    """Compare prompt-failure rates between the two onboarding variants.

    Uses a two-sided two-proportion z-test (pooled under H0) on the session-level
    prompt-failure counts.

    Confidence interval: 95% Wald CI for the difference in proportions
    (guided_tutorial - self_serve).
    """
    required = {
        "onboarding_variant",
        "sessions_with_queries",
        "sessions_first_query_prompt_failures",
        "session_prompt_failure_rate",
    }
    missing = required - set(funnel_df.columns)
    if missing:
        raise ValueError(f"funnel_df missing required columns: {sorted(missing)}")

    def row_for(variant: str) -> pd.Series:
        rows = funnel_df[funnel_df["onboarding_variant"] == variant]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly 1 row for onboarding_variant={variant!r}, got {len(rows)}")
        return rows.iloc[0]

    self_row = row_for("self_serve")
    guided_row = row_for("guided_tutorial")

    n_self = int(self_row["sessions_with_queries"])
    x_self = int(self_row["sessions_first_query_prompt_failures"])
    p_self = float(self_row["session_prompt_failure_rate"])

    n_guided = int(guided_row["sessions_with_queries"])
    x_guided = int(guided_row["sessions_first_query_prompt_failures"])
    p_guided = float(guided_row["session_prompt_failure_rate"])

    if n_self <= 0 or n_guided <= 0:
        raise ValueError("Denominators must be > 0 for both variants")

    # Difference is guided - self
    diff = p_guided - p_self

    # Two-proportion z test with pooled standard error under H0
    p_pool = (x_self + x_guided) / (n_self + n_guided)
    se_pool = (p_pool * (1.0 - p_pool) * (1.0 / n_self + 1.0 / n_guided)) ** 0.5
    if se_pool == 0:
        # Degenerate case (both rates 0 or 1). Return CI at diff and p=1.
        p_value = 1.0
        ci_low = diff
        ci_high = diff
    else:
        z = diff / se_pool
        p_value = 2.0 * norm.sf(abs(z))

        # Wald 95% CI for diff using unpooled SE
        se_unpooled = (p_self * (1.0 - p_self) / n_self + p_guided * (1.0 - p_guided) / n_guided) ** 0.5
        ci_low = diff - 1.96 * se_unpooled
        ci_high = diff + 1.96 * se_unpooled

    # Effect size: absolute difference (percentage points) and relative change
    effect_abs = diff
    effect_rel = diff / p_self if p_self != 0 else float("inf")

    return {
        "p_prompt_failure_self_serve": p_self,
        "p_prompt_failure_guided_tutorial": p_guided,
        "absolute_effect_guided_minus_self": effect_abs,
        "relative_effect_guided_vs_self": effect_rel,
        "p_value_two_sided": p_value,
        "ci95_low_absolute_effect": ci_low,
        "ci95_high_absolute_effect": ci_high,
        "n_sessions_self_serve": n_self,
        "x_prompt_fail_self_serve": x_self,
        "n_sessions_guided_tutorial": n_guided,
        "x_prompt_fail_guided_tutorial": x_guided,
    }


def write_conclusion(funnel_df: pd.DataFrame, stats_result: Dict[str, float]) -> str:
    """Return a decision-facing plain-language conclusion.

    Interprets the stakeholder headline as:
      "guided_tutorial reduces prompt-failure compared to self_serve".
    """

    def pct(x: float) -> str:
        if x == float("inf"):
            return "inf"
        return f"{100.0 * x:.2f}%"

    self_row = funnel_df[funnel_df["onboarding_variant"] == "self_serve"].iloc[0]
    guided_row = funnel_df[funnel_df["onboarding_variant"] == "guided_tutorial"].iloc[0]

    p_self = stats_result["p_prompt_failure_self_serve"]
    p_guided = stats_result["p_prompt_failure_guided_tutorial"]
    diff = stats_result["absolute_effect_guided_minus_self"]
    p_value = stats_result["p_value_two_sided"]
    ci_low = stats_result["ci95_low_absolute_effect"]
    ci_high = stats_result["ci95_high_absolute_effect"]

    answered_query_rate_self = float(self_row["answered_query_rate"])
    answered_query_rate_guided = float(guided_row["answered_query_rate"])
    share_sessions_answered_self = float(self_row["share_sessions_with_answered"])
    share_sessions_answered_guided = float(guided_row["share_sessions_with_answered"])

    supports = (p_value < 0.05) and (diff < 0)
    # Also consider "directional" evidence even if not significant.
    direction = "lower" if diff < 0 else "higher" if diff > 0 else "the same"

    if supports:
        headline = "supports" 
        lead = (
            f"The segmented analysis supports the headline that the guided_tutorial experience reduces "
            f"first-query prompt failures. "
        )
    elif p_value < 0.05 and diff > 0:
        headline = "challenges"
        lead = (
            f"The segmented analysis challenges the headline: guided_tutorial has a statistically higher "
            f"first-query prompt-failure rate than self_serve. "
        )
    else:
        headline = "does not clearly support"
        lead = (
            f"The segmented analysis does not clearly support the headline. "
            f"(Directionally, guided_tutorial has a {direction} first-query prompt-failure rate, but the "
            f"difference is not statistically significant at the 0.05 level.) "
        )

    return (
        f"{lead}"
        f" First-query prompt-failure rate (session-level; 'no_prompt_entered') was {pct(p_guided)} for guided_tutorial "
        f"vs {pct(p_self)} for self_serve, a difference of {100.0 * diff:.2f} percentage points "
        f"(95% CI [{100.0 * ci_low:.2f} pp, {100.0 * ci_high:.2f} pp], two-sided p={p_value:.4g}). "
        f"For context, answered-query rate was {pct(answered_query_rate_guided)} for guided_tutorial vs {pct(answered_query_rate_self)} for self_serve, "
        f"and the share of sessions with at least one answered query was {pct(share_sessions_answered_guided)} vs {pct(share_sessions_answered_self)}. "
        f"Conclusion: based on this segmented (non-causal) funnel comparison, the evidence {headline} the naive headline, "
        f"so any product claim should be framed as an association rather than proof of causal impact."
    )


def main():
    with get_connection() as conn:
        sanity = pd.read_sql_query(
            "SELECT COUNT(*) AS query_rows, COUNT(DISTINCT session_id) AS query_bearing_sessions FROM assistant_queries",
            conn,
        )
        print("Database connection OK")
        print(sanity.to_string(index=False))
        try:
            funnel = build_segment_funnel(conn)
            stats_result = run_significance_test(funnel)
            conclusion = write_conclusion(funnel, stats_result)
            print(funnel.to_string(index=False))
            print(stats_result)
            print(conclusion)
        except NotImplementedError as exc:
            print(f"Scaffold incomplete: {exc}")


if __name__ == "__main__":
    main()
