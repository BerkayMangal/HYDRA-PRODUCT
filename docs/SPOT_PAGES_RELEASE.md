# Standalone spot preview publication

The Pages environment permits deployment from `main`. The Pages workflow is
therefore defined on `main` and checks out the reviewed UI source revision from
`HYDRA_SPOT_SOURCE_SHA`. It builds and tests that revision and verifies its commit
through the published `/spot/release.json` file. No user signing or exchange POST
occurs in the workflow.

This commit changes only publication configuration and this document. The live
Railway application includes model routes absent from this checkout. Do not
replace it with this older main application as part of publishing the static UI.

Commits dedicated to this static publication start with `[spot-preview]`.
The Railway workflow explicitly excludes that commit prefix from automatic
`workflow_run` deployments; Python CI still runs normally. Manual Railway
releases and ordinary application commits retain their existing conditions.

The Pages environment protection is kept: the publishing workflow runs from
`main`. The UI source remains pinned to the already reviewed feature revision.

Expected phone URL after successful verification:
https://berkaymangal.github.io/HYDRA-PRODUCT/spot/

The first phone trial uses the wallet's built-in browser for connection, balances
and a buy preview. Real wallet compatibility and actual fills are not established
by mocked tests. The screen requests a separate user signature for each buy;
it does not authorize an API agent or charge a builder fee.
