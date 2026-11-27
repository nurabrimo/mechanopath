source dcc/config.sh
source activate $CONDA_ENV
python src/13_mechanics_phase2.py && python src/14_design_plasmode.py --reps 20 --grid-reps 5 && python dcc/08_figures.py fig5 fig6 ed3 ed4 ed5
