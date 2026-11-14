# Security

This repository contains benchmark scripts and data, not a service. The parts that touch credentials are the
RunPod helpers in `tools/`. They read `RUNPOD_API_KEY` from the environment, send it only to
`api.runpod.io` over HTTPS, and never print or store it.

If you find a way these scripts could leak a credential, or anything else with security impact, please report
it privately with GitHub's **Report a vulnerability** button under the Security tab, not in a public issue.
