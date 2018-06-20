########### TEXTURE MODEL CONFIG ######################################################

# use between filterpyramids: False - filterbank; True: filterbank_simoncelli
FILTERPYRAMID = True
# spatial neighborhood is Na x Na coefficients (must be odd)
NA = 9

########### SYNTHESIS CONFIG ######################################################
# Number of iterations of synthesis procedure is only set if no certain NITER is given in application
NITERATION = 12
# desired shape of the synthesized image - only set if no shape is given.
SHAPE = (512,512)


########### APPLICATION CONFIG ######################################################
# set mask to keep (right) part of image
MASK = False

########### CREATING DEFECT CONFIG ######################################################
# for defect creation/generation - 0: line defect
DEFECT_CLASS = 0

# are prints fully covered? (True) or is paper visible in images? (False)
FULLY_COVERED = True

# relative intensity
# True: for percentual (transparent) intensity values between actual value and 1
# False: for covering defect intensity
REL_INTENSITY = True

########### REDUCTION CONFIG ######################################################

# margin defines the factor which is multiplied to (original - reference - tolerance)
# for each parameter as additional noise
MARGIN = 1

# set absolute tolerance which
ATOL = 1.e-5

########### DEBUG CONFIG ######################################################

#use initial seed (same as in matlab)
SEED = False

# save correlation matrices as image in result/.
PLOT_CORR = False

# save parameters from python analysis to load in matlab and use matlab synthesis
SAVE_PARAMS = False

# compare coefficients with matlab coefficients (you need to compute matlab coeff first)
COMPARE_COEFF = False

# compare parameters to matlab parameters (you need to compute matlab params first)
COMPARE_TO_MAT = False

#matlab_path = ABSOLUTE_PATH + '/matlab'
#result_path = ABSOLUTE_PATH + '/project/result'

