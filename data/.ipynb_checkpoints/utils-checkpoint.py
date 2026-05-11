import os
import os.path
import copy
import hashlib
import errno
import numpy as np
from numpy.testing import assert_array_almost_equal


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
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

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )
    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return directories


def list_files(root, suffix, prefix=False):
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=21):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    print(P)
    n = noise

    if n > 0.0:
        if nb_classes == 21:
            for i in range(nb_classes):
                P[i, i] = 1. - n
            P[0, 0] += n
            P[1, 17] += n    # Airplane -> Runway
            P[2, 9] += n     # Baseball Diamond -> Golf Course
            P[3, 4] += n     # Beach -> Buildings
            P[4, 4] += n
            P[5, 5] += n
            P[6, 12] += n    # Dense Residential -> Medium Residential
            P[7, 7] += n
            P[8, 11] += n    # Freeway -> Intersection
            P[9, 20] += n    # Golf Course -> Tennis Court
            P[10, 16] += n   # Harbor -> River
            P[11, 11] += n
            P[12, 12] += n
            P[13, 18] += n   # Mobile Home Park -> Sparse Residential
            P[14, 8] += n    # Overpass -> Freeway
            P[15, 19] += n   # Parking Lot -> Storage Tanks
            P[16, 16] += n
            P[17, 17] += n
            P[18, 18] += n
            P[19, 19] += n
            P[20, 20] += n
        elif nb_classes == 30:
            for i in range(nb_classes):
                P[i, i] = 1. - n
            P[0, 1] += n     # Airport -> Bare Land
            P[1, 1] += n
            P[2, 18] += n    # Baseball Field -> Playground
            P[3, 1] += n     # Beach -> Bare Land
            P[4, 29] += n    # Bridge -> Viaduct
            P[5, 5] += n
            P[6, 6] += n
            P[7, 7] += n
            P[8, 14] += n    # Dense Residential -> Medium Residential
            P[9, 9] += n
            P[10, 10] += n
            P[11, 13] += n   # Forest -> Meadow
            P[12, 12] += n
            P[13, 13] += n
            P[14, 14] += n
            P[15, 15] += n
            P[16, 18] += n   # Park -> Playground
            P[17, 28] += n   # Parking Lot -> Storage Tanks
            P[18, 18] += n
            P[19, 19] += n
            P[20, 21] += n   # Port -> Railway Station
            P[21, 21] += n
            P[22, 22] += n
            P[23, 23] += n
            P[24, 26] += n   # School -> Square
            P[25, 25] += n
            P[26, 26] += n
            P[27, 27] += n
            P[28, 28] += n
            P[29, 29] += n
        else:
            # sequential pair flip for any other number of classes
            P[0, 0], P[0, 1] = 1. - n, n
            for i in range(1, nb_classes - 1):
                P[i, i], P[i, i + 1] = 1. - n, n
            P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
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
        P[0, 0] += 1. - n
        for i in range(1, nb_classes - 1):
            P[i, i] += 1. - n
        P[nb_classes - 1, nb_classes - 1] += 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise


def noisify(dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate