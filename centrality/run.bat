@echo off
REM Activate your conda environment
call conda activate centrality

REM Define lists
@REM set datasets=senate-bills house-bills email-enron email-eu contact-primary-school contact-high-school tags-ask-ubuntu tags-math-sx coauth-mag-history coauth-dblp
set datasets=email-enron
set node_centralities=harmonic
set edge_centralities=harmonic

REM Loop over datasets
for %%D in (%datasets%) do (
    @REM REM Loop over node centralities
    @REM for %%C in (%node_centralities%) do (
    @REM     python test_single_centrality.py --dataset %%D --measure %%C
    @REM )

    REM Loop over edge centralities
    for %%C in (%edge_centralities%) do (
        python test_single_centrality.py --dataset %%D --measure %%C --edge
    )
)
