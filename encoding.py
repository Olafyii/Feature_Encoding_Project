from sklearn.cluster import KMeans
import numpy as np
import os
from sklearn.mixture import GaussianMixture
import time
import math
import argparse
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', type=float)
parser.add_argument('--codebook_size', type=int)
args = parser.parse_args()

npy_root_path = '/home/google/SIFT/sift_descriptors_10pcnt'
output_path = '/home/google/SIFT/encoded_features/'
all_proposals = np.load('/home/google/SIFT/train_set_proposals.npy')
train_img_txt = '/home/google/SIFT/train_set.txt'
test_img_txt = '/home/google/SIFT/test_set.txt'
log_path = '/home/google/SIFT/log.txt'

# os.system('mkdir -p '+output_path+'_ratio'+str(args.ratio)+'_cb'+str(args.codebook_size)+'/BOW')
# os.system('mkdir -p '+output_path+'_ratio'+str(args.ratio)+'_cb'+str(args.codebook_size)+'/VLAD')
# os.system('mkdir -p '+output_path+'_ratio'+str(args.ratio)+'_cb'+str(args.codebook_size)+'/Fisher')

class_name = []
for root, dirs, files in os.walk('/home/google/proposals/Proposals_txt_414'):
    if len(dirs) != 0:
        class_name = dirs
dictionary = {name:label for name, label in zip(sorted(class_name), range(len(class_name)))}

def encoder(all_proposals, codebook_size, sample_ratio, do_encode, feature_dim, gmm=None):
    os.system('mkdir -p '+output_path+'_ratio'+str(args.ratio)+'_cb'+str(args.codebook_size)+'/BOW')
    os.system('mkdir -p '+output_path+'_ratio'+str(args.ratio)+'_cb'+str(args.codebook_size)+'/VLAD')
    os.system('mkdir -p '+output_path+'_ratio'+str(args.ratio)+'_cb'+str(args.codebook_size)+'/Fisher')
    if type(all_proposals) == type(0):
        print('Loading npy')
        all_proposals = np.load('/home/google/small_dataset/train_set_proposals.npy')
        print('Loading finished. Proposal size:', all_proposals.shape)

    t0 = time.time()
    print('Sampling')
    index = np.random.choice(all_proposals.shape[0], size=int(all_proposals.shape[0]*sample_ratio), replace=False)
    sampled_proposal = all_proposals[index, :]  # sample
    t1 = time.time()
    print('Sampling finished. Size: %d. Time: %.5f\n' % (sampled_proposal.shape[0], t1-t0))
    # log = open(log_path, 'a')
    # log.write('Sampling finished. Size: %d. Time: %.5f\n' % (int(all_proposals.shape[0]*sample_ratio), t1-t0))
    # log.write('\n')
    # log.close()

    if not gmm:
        print('GMMing')
        gmm = GaussianMixture(n_components=codebook_size, verbose=1, covariance_type='diag').fit(sampled_proposal)
        t2 = time.time()
        print('GMM finished. Time: %.5f\n' % (t2-t1))
        log = open(log_path, 'a')
        log.write('GMM finished. Time: %.5f' % (t2-t1))
        log.write('\n')
        log.close()
        if not do_encode:
            print('GMM is returned.')
            return gmm
    t2 = time.time()

    print('Encoding')
    cnt = 0
    # delta_a, delta_b, delta_c, delta_d, delta_e = 0, 0, 0, 0, 0
    for root, dirs, files in os.walk(npy_root_path):
        if len(dirs) == 0:
            for f in files:
                if os.path.isfile(output_path+'_ratio'+str(args.ratio)+'_cb'+str(args.codebook_size)+'/Fisher/'+f):
                    print('pass')
                    continue
                proposals = np.load(root+'/'+f)
                if len(proposals.shape) == 1:
                    proposals = proposals.reshape((1,feature_dim))

                encoded_feature_BOW = np.zeros(codebook_size)
                encoded_feature_VLAD = np.zeros(codebook_size*feature_dim)
                encoded_feature_Fisher = np.zeros(codebook_size*feature_dim*2)

                for i in range(proposals.shape[0]):
                    ta = time.time()
                    proba = gmm.predict_proba(proposals[[i]]).flatten()
                    tb = time.time()
                    # delta_a += (tb-ta)
                    for t in range(codebook_size):
                        # k = gmm.predict(proposals[[i]])[0]
                        ta = time.time()
                        gamma_numerator = gmm.weights_[t]*proba[t]
                        gamma_denominator = (gmm.weights_*proba).sum()
                        tc = time.time()
                        # delta_b += (tc-ta)
                        if gamma_denominator == 0:
                            gamma = 0
                        else:
                            gamma = gamma_numerator / gamma_denominator
                        encoded_feature_BOW[t] += gamma
                        ta = time.time()
                        delta = proposals[i] - gmm.means_[t]
                        denominator = (proposals.shape[0]*math.sqrt(gmm.weights_[t]))
                        vlad = (gamma * (delta) / gmm.covariances_[t]) / denominator
                        tb = time.time()
                        # delta_c += (tb-ta)
                        fisher = (gamma * (np.power(delta, 2)/np.power(gmm.covariances_[t],2)-1))/denominator*math.sqrt(2)
                        tc = time.time()
                        # delta_d += (tc-tb)
                        encoded_feature_VLAD[t*feature_dim:t*feature_dim+feature_dim] += vlad
                        encoded_feature_Fisher[t*feature_dim*2:t*feature_dim*2+feature_dim] += vlad
                        encoded_feature_Fisher[t*feature_dim*2+feature_dim:t*feature_dim*2+feature_dim*2] += fisher
                        td = time.time()
                        # delta_e += (td-tc)
                # print(encoded_feature_BOW.shape, encoded_feature_VLAD.shape, encoded_feature_Fisher.shape)
                np.save(output_path+'_ratio'+str(args.ratio)+'_cb'+str(args.codebook_size)+'/Fisher/'+f, encoded_feature_Fisher)
                np.save(output_path+'_ratio'+str(args.ratio)+'_cb'+str(args.codebook_size)+'/VLAD/'+f, encoded_feature_VLAD)
                np.save(output_path+'_ratio'+str(args.ratio)+'_cb'+str(args.codebook_size)+'/BOW/'+f, encoded_feature_BOW)

                if cnt % 5 == 0:
                    print(cnt)
                cnt += 1
    t3 = time.time()
    print('Encoding finished. Time: %.5f\n' % (t3-t2))
    log = open(log_path, 'a')
    log.write('Encoding finished. Time: %.5f' % (t3-t2))
    log.write('\n')
    log.close()

    # encoded_BOW = encoded_BOW[:cnt, :]
    return gmm



def pca_reduction(folder, n, feature_dim, codebook_size):
    flag = True
    for key, fd in [('VLAD', feature_dim*codebook_size), ('Fisher', feature_dim*codebook_size*2)]:
        print('PCA ', key)
        log = open(log_path, 'a')
        log.write('PCA '+key+'\n')
        log.close()
        if not os.path.isdir(folder+'/PCA'+str(n)+'_'+key):
            os.system('mkdir -p '+folder+'/PCA'+str(n)+'_'+key)
        print('loading all encoded features to one npy file')
        t0 = time.time()
        f = open(train_img_txt)
        files = f.readlines()
        # files = files[:100]
        f.close()
        for i in range(len(files)):
            if files[i][-1] == '\n':
                files[i] = folder+'/'+key+'/'+files[i][:-1]
            else:
                files[i] = folder+'/'+key+'/'+files[i]
        print(len(files), feature_dim*codebook_size)
        _all = np.zeros((len(files), fd))
        for idx, f in enumerate(files):
            # print(a.shape)
            if flag:
                print('PCA using:', f)
                flag = False
            _all[idx, :] = np.load(f)
            if idx % 500 == 0:
                print(idx)

        t1 = time.time()
        print('loading all encoded features to one npy file FINISHED, time: %d' % (t1-t0))
        # log = open(log_path, 'a')
        # log.write('loading all encoded features to one npy file FINISHED, time: %d' % (t1-t0))
        # log.write('\n')
        # log.close()

        print('doing PCA, n= %d' % n)
        t0 = time.time()
        pca = PCA(n_components=n)
        pca.fit(_all)
        t1 = time.time()
        print('PCA FINISHED, time: %d' % (t1-t0))
        log = open(log_path, 'a')
        log.write('PCA FINISHED, time: %d' % (t1-t0))
        log.write('PCA n = %d' % n)
        log.write('\n')
        log.close()

        print('transforming')
        t0 = time.time()
        for cnt, f in enumerate(glob.glob(folder+'/'+key+'/*.npy')):
            a=np.load(f)
            a=pca.transform([a])
            np.save(f.split(key)[0]+'PCA'+str(n)+'_'+key+f.split(key)[1], a)
            if cnt % 500 == 0:
                print(cnt)
        t1 = time.time()
        print('transform FINISHED, time: %d' % (t1-t0))
        log = open(log_path, 'a')
        log.write('transform FINISHED, time: %d' % (t1-t0))
        log.write('\n')
        log.close()
    return _all



def svm(folder, codebook_size, feature_dim, n):
    flag = True
    for key, fd in [('BOW', codebook_size), ('PCA'+str(n)+'_VLAD', n), ('PCA'+str(n)+'_Fisher', n)]:
        f = open(train_img_txt)
        train_files = f.readlines()
        # train_files = train_files[:1000]
        f.close()
        f = open(test_img_txt)
        test_files = f.readlines()
        # test_files = test_files[:1000]
        f.close()
        for i in range(len(train_files)):
            if train_files[i][-1] == '\n':
                train_files[i] = folder+'/'+key+'/'+train_files[i][:-1]
            else:
                train_files[i] = folder+'/'+key+'/'+train_files[i]
        for i in range(len(test_files)):
            if test_files[i][-1] == '\n':
                test_files[i] = folder+'/'+key+'/'+test_files[i][:-1]
            else:
                test_files[i] = folder+'/'+key+'/'+test_files[i]
        # train_files = train_files[10000:11000]
        X = np.zeros((len(train_files), fd))
        y = np.zeros(len(train_files))
        X_test = np.zeros((len(test_files), fd))
        y_test = np.zeros(len(test_files))
        t0 = time.time()
        print('making training samples')
        for idx, f in enumerate(train_files):
            # print(f)
            if flag and key[-1] == 'D':
                print('SVM using:', f)
                flag = False
            X[idx, :] = np.load(f)
            y[idx] = dictionary[f.split('/')[-1].split('_')[0]]
            if idx%1000==0:
                print(idx)
        for idx, f in enumerate(test_files):
            # print(f)
            X_test[idx, :] = np.load(f)
            y_test[idx] = dictionary[f.split('/')[-1].split('_')[0]]
            if idx%500==0:
                print(idx)
        t1 = time.time()
        print(X.shape)
        print(y.shape)
        print('making training samples FINISHED, time: %d' % (t1-t0))
        # log = open(log_path, 'a')
        # log.write('making training samples FINISHED, time: %d' % (t1-t0))
        # log.write('\n')
        # log.close()
        clf = SVC(gamma='auto')
        t0 = time.time()
        clf.fit(X, y)
        t1 = time.time()
        print('SVM fit FINISHED, time: %d\n' % (t1-t0))
        log = open(log_path, 'a')
        log.write('SVM fit FINISHED, time: %d\n' % (t1-t0))
        log.close()
        t0 = time.time()
        acc = clf.score(X_test, y_test)
        t1 = time.time()
        log = open(log_path, 'a')
        log.write('SVM score FINISHED, time: %d\n' % (t1-t0))
        log.close()
        print('codebook_size '+str(args.codebook_size)+' ratio '+str(args.ratio)+' '+key+'PCA n = ' + str(n) + ' acc '+str(acc))
        log = open(log_path, 'a')
        log.write('codebook_size '+str(args.codebook_size)+' ratio '+str(args.ratio)+' '+key+'PCA n = ' + str(n) + ' acc '+str(acc)+'\n')
        log.close()



log = open(log_path, 'a')
log.write('codebook_size: '+str(args.codebook_size)+' ratio: '+str(args.ratio))
log.write('\n')
log.close()
gmm=encoder(all_proposals, args.codebook_size, args.ratio, do_encode=True, feature_dim=128)
pca_reduction(output_path+'_ratio'+str(args.ratio)+'_cb'+str(args.codebook_size), n=args.codebook_size, feature_dim=128, codebook_size=args.codebook_size)
svm(output_path+'_ratio'+str(args.ratio)+'_cb'+str(args.codebook_size), feature_dim=128, codebook_size=args.codebook_size, n=args.codebook_size)
log = open(log_path, 'a')
log.write('\n')
log.close()
