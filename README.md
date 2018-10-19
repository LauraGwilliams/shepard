# shepard
stimulus creation and analysis scripts for shepard tone experiment


# NOTES:
* stimuli are first made using the python notebook; this makes the stimuli at around 70 db. then we normalise them fully in praat.

# TODO:
- see if we can figure out the computation that makes the db match in the notebook, without doing the additional praat step.
- the db of the shepards is not consistent within a scale ... is this by design, or not? check with diana.
- try reducing more the lowest amplitude -- it should be 37Db, but make it quiter and see if this averages out the overall amp? and also this will make the transition smoother from step to step.