# Solution Steps

1. In `analysis/analyze.py`, implement `build_segment_funnel(conn)` by writing a single SQL query that aggregates by `user_onboarding.onboarding_variant` and produces (i) session-level first-query prompt-failure rate, (ii) answered-query rate, and (iii) share of sessions with at least one answered query.

2. In the SQL, avoid double-counting sessions by first deriving one row per `assistant_sessions.session_id` for the *first* query (earliest `query_timestamp`) within a CTE (e.g., using `DISTINCT ON (session_id)` ordered by timestamp). Use this CTE to compute the session-level prompt-failure numerator/denominator.

3. Compute the answered-query rate from query rows directly (`COUNT(*)` and `SUM(response_status='answered')`) in a separate CTE, and compute the session-answered share by collapsing to one row per session with a boolean aggregate like `BOOL_OR(response_status='answered')`.

4. Join the resulting per-variant CTEs in the final `SELECT` so you get exactly one row per onboarding variant with all required counts and rates.

5. Next, implement `run_significance_test(funnel_df)` to compare the two variants’ *session-level* prompt-failure rates using a two-proportion two-sided z-test: numerator = first-query prompt failures, denominator = sessions with queries.

6. Compute a 95% confidence interval for the difference in proportions (guided_tutorial − self_serve) using a Wald CI with unpooled standard error; return p-value, CI bounds, absolute effect size (difference in rates), and relative effect size (difference divided by self_serve rate).

7. Finally, implement `write_conclusion(funnel_df, stats_result)` to translate the numbers into business language: state the observed prompt-failure rates by variant, whether the CI/p-value support the stakeholder headline (assumed claim: guided_tutorial reduces prompt failures vs self_serve), and add a plain-language note that this is observational funnel evidence, not causal proof.

8. Run `./run.sh`, then execute `python analysis/analyze.py` to confirm the SQL returns both onboarding variants and that the computed statistics and conclusion print successfully.

