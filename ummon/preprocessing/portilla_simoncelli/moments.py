import numpy as np

def skew(img):
	mn = np.mean(img)
	var = np.var(img)
	return skew2(img, mn, var)

def skew2(img, mn, var):
	if np.isrealobj(img):
		#return np.mean((img.astype(np.uint32)-mn)**3) / (var**(3/2))
		return np.mean((img-mn)**3) / (var**(3/2))
	else:
		#return complex(np.mean(np.real(img.astype(np.uint32)-mn)**3) / np.real(var)**(3/2), np.mean(np.imag(img.astype(np.uint32)-mn)**3) / (np.imag(var) ** (3/2)))
		return complex(np.mean(np.real(img-mn)**3) / np.real(var)**(3/2), np.mean(np.imag(img-mn)**3) / (np.imag(var) ** (3/2)))


def kurt(img):
	mn = np.mean(img)
	var = np.var(img)
	return kurt2(img, mn, var)

def kurt2(img, mn, var):
	if np.isrealobj(img):
		#return np.mean(np.absolute(img.astype(np.uint32)-mn)**4) / (var**2)
		return np.mean(np.absolute(img-mn)**4) / (var**2)
	else:
		#return complex(np.mean(np.real(img.astype(np.uint32)-mn)**4) / np.real(var)**(3/2), np.mean(np.imag(img.astype(np.uint32)-mn)**4) / (np.imag(var)**2))
		return complex(np.mean(np.real(img-mn)**4) / np.real(var)**(3/2), np.mean(np.imag(img-mn)**4) / (np.imag(var)**2))
