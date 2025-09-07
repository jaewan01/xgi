@echo off
REM Activate your conda environment
call conda activate centrality

REM Define lists
@REM set datasets=senate-bills house-bills email-enron email-eu contact-primary-school contact-high-school tags-ask-ubuntu tags-math-sx coauth-mag-history coauth-dblp
set datasets=senate-bills
@REM set node_centralities=eigenvector
@REM set node_centralities=degree neighbor_degree closeness betweenness harmonic eigenvector pagerank uplift_eigenvector hypercoreness
@REM set edge_centralities=degree line_expansion_degree closeness betweenness harmonic eigenvector pagerank hypercoreness
set edge_centralities=eigenvector pagerank hypercoreness

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
