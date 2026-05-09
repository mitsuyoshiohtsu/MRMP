import errno
import hashlib
import os
import os.path

import numpy as np
from numpy.testing import assert_array_almost_equal


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, "rb") as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:
        try:
            print("Downloading " + url + " to " + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print("Failed download. Trying https -> http instead." " Downloading " + url + " to " + fpath)
                urllib.request.urlretrieve(url, fpath)






# basic function
def multiclass_noisify(y, P, random_state=0):
    """Flip classes according to transition probability matrix T.

    It expects a number between 0 and the number of classes - 1.
    """
    print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
    flip in the pair
    """
    P = np.eye(nb_classes)
    print(P)
    n = noise

    if n > 0.0:
        if nb_classes == 10:
            for i in range(nb_classes):
                P[i, i] = 1.0 - n
            P[0, 0] += n
            P[2, 0] += n
            P[4, 7] += n
            P[7, 7] += n
            P[1, 1] += n
            P[9, 1] += n
            P[3, 5] += n
            P[5, 3] += n
            P[6, 6] += n
            P[8, 8] += n
        else:
            # 0 -> 1
            P[0, 0], P[0, 1] = 1.0 - n, n
            for i in range(1, nb_classes - 1):
                P[i, i], P[i, i + 1] = 1.0 - n, n
            P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1.0 - n, n

        ## use simulated pairs

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print("Actual noise %.2f" % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
    flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / nb_classes) * P

    if n > 0.0:
        P[0, 0] += 1.0 - n
        for i in range(1, nb_classes - 1):
            P[i, i] += 1.0 - n
        P[nb_classes - 1, nb_classes - 1] += 1.0 - n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print("Actual noise %.2f" % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise


def noisify(
    dataset="mnist",
    nb_classes=10,
    train_labels=None,
    noise_type=None,
    noise_rate=0,
    random_state=0,
):
    if noise_type == "pairflip":
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == "symmetric":
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate