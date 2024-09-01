import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.svm import LinearSVC

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# X_train has 32 columns containing the challeenge bits
	# y_train contains the responses
	regr = LinearSVC(loss='squared_hinge', penalty='l2', dual=False)
	X=my_map(X_train)
	# print("Mapped")
	regr.fit(X, y_train)
	w=regr.coef_[0]
	b=regr.intercept_[0]
	# print(w, b)
	return w.T, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
	# Vectorized innermap
    def innermap(c):
        d = 2*c - 1
        return np.flip(np.cumprod(d[::-1]))

    # Vectorized innermap
    x = np.apply_along_axis(innermap, 1, X)

    n_dims = x.shape[1]

    # Quadratic terms
    comb = np.triu_indices(n_dims, 1)
    x2 = x[:, comb[0]] * x[:, comb[1]]

    num_quadratic = int((n_dims * (n_dims - 1)) / 2)

    # Initialize output
    num_samples = len(X)
    feat = np.zeros((num_samples, num_quadratic + n_dims), dtype=int)

    # Populate quadratic terms
    feat[:, :num_quadratic] = x2

    # Populate linear terms
    feat[:, num_quadratic:] = x

    return feat
