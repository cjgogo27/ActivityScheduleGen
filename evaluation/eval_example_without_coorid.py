## file: eval
"""evaluation"""
import numpy as np
import editdistance
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu
from scipy.spatial import distance
from nltk.translate.bleu_score import SmoothingFunction

smoothie = SmoothingFunction().method1

def acc(gen_seq, tar_seq):
    return np.sum(gen_seq == tar_seq) / (gen_seq.shape[0] * gen_seq.shape[1])

def f1(gen_seq, tar_seq):
    return f1_score(tar_seq.reshape(-1), gen_seq.reshape(-1), average='macro')

def edit_dist(gen_seq, tar_seq):
    edit_dist_list = []
    for i in range(tar_seq.shape[0]):
        tar_sequence = [str(k) for k in tar_seq[i].tolist()]
        gen_sequence = [str(k) for k in gen_seq[i].tolist()]
        edit_dist = editdistance.eval(tar_sequence, gen_sequence) / len(tar_sequence)
        edit_dist_list.append(edit_dist)
    return np.mean(edit_dist_list)


def bleu_score(gen_seq, tar_seq):
    bleu_score_list = []
    for i in range(tar_seq.shape[0]):
        tar_sequence = [str(k) for k in tar_seq[i].tolist()]
        gen_sequence = [str(k) for k in gen_seq[i].tolist()]
        try:
            bleu_sc = sentence_bleu([tar_sequence], gen_sequence, smoothing_function=smoothie)
        except Exception as e:
            try:
                bleu_sc = sentence_bleu([tar_sequence], gen_sequence)
            except:
                matches = sum(1 for a, b in zip(tar_sequence, gen_sequence) if a == b)
                bleu_sc = matches / len(tar_sequence)
        bleu_score_list.append(bleu_sc)
    return np.mean(bleu_score_list)

def dataset_jsd(gen_seq, tar_seq):
    test_trajs_str = ['_'.join([str(k) for k in tar_seq[i].tolist()]) for i in range(len(tar_seq))]
    test_trajs_set = set(test_trajs_str)
    test_trajs_dict = dict(zip(list(test_trajs_set), range(len(test_trajs_set))))
    test_trajs_label = [test_trajs_dict[traj] for traj in test_trajs_str]
    test_trajs_label.append(0)
    test_p = np.histogram(test_trajs_label)[0] / len(test_trajs_label)

    pad_idx = len(test_trajs_set)
    learner_trajs_str = ['_'.join([str(k) for k in gen_seq[i].tolist()]) for i in range(len(gen_seq))]
    learner_trajs_label = [test_trajs_dict.get(traj, pad_idx) for traj in learner_trajs_str]
    learner_p = np.histogram(learner_trajs_label)[0] / len(learner_trajs_label)
    return distance.jensenshannon(test_p, learner_p)

def compute_int(act_seq, n_time):
    print("act_seq", act_seq.shape)
    act2int = np.zeros((11, n_time)) # count of intervals of different activities
    for i in range(act_seq.shape[0]):
        curr_act, curr_int = act_seq[i, 0], 1
        for j in range(1, act_seq.shape[1]):
            if act_seq[i, j] == curr_act:
                curr_int += 1
            else:
                act2int[curr_act, curr_int - 1] = act2int[curr_act, curr_int - 1] + 1
                curr_act, curr_int = act_seq[i, j], 1
        act2int[curr_act, curr_int - 1] = act2int[curr_act, curr_int - 1] + 1
    return act2int

def macro_micro_int_jsd(gen_seq, tar_seq, n_time):
    gen_act2int = compute_int(gen_seq, n_time)
    tar_act2int = compute_int(tar_seq, n_time)
    macro_int_jsd = distance.jensenshannon(np.sum(gen_act2int, 0) / np.sum(gen_act2int), np.sum(tar_act2int, 0) / np.sum(tar_act2int))
    micro_int_jsd = distance.jensenshannon(gen_act2int.reshape(-1) / np.sum(gen_act2int), tar_act2int.reshape(-1) / np.sum(tar_act2int))
    return macro_int_jsd, micro_int_jsd

def compute_act_type(act_seq):
    act2cnt = np.zeros(11)
    for i in range(11):
        act2cnt[i] = np.sum(act_seq == i)
    return act2cnt

def act_type_jsd(gen_seq, tar_seq):
    gen_act2cnt = compute_act_type(gen_seq)
    tar_act2cnt = compute_act_type(tar_seq)
    type_jsd = distance.jensenshannon(gen_act2cnt / np.sum(gen_act2cnt), tar_act2cnt / np.sum(tar_act2cnt))
    return type_jsd

def compute_uni_act_type(act_seq):
    act2cnt = np.zeros(11)
    for i in range(act_seq.shape[0]):
        curr_act = act_seq[i, 0]
        act2cnt[curr_act] = act2cnt[curr_act] + 1
        for j in range(1, act_seq.shape[1]):
            if act_seq[i, j] == curr_act:
                continue
            else:
                curr_act = act_seq[i, j]
                act2cnt[curr_act] = act2cnt[curr_act] + 1
    return act2cnt

def uni_act_type_jsd(gen_seq, tar_seq):
    gen_act2cnt = compute_uni_act_type(gen_seq)
    tar_act2cnt = compute_uni_act_type(tar_seq)
    type_jsd = distance.jensenshannon(gen_act2cnt / np.sum(gen_act2cnt), tar_act2cnt / np.sum(tar_act2cnt))
    return type_jsd

def compute_traj_len(act_seq):
    traj_len_ls = []
    for i in range(act_seq.shape[0]):
        curr_act = act_seq[i, 0]
        traj_len = 1
        for j in range(1, act_seq.shape[1]):
            if act_seq[i, j] == curr_act:
                continue
            else:
                curr_act = act_seq[i, j]
                traj_len += 1
        traj_len_ls.append(traj_len)
    traj_len_array = np.array(traj_len_ls)
    traj_len_dist = np.zeros(np.max(traj_len_array))
    for i in range(len(traj_len_dist)):
        traj_len_dist[i] = np.sum(traj_len_array == i+1)
    return traj_len_dist

def traj_len_jsd(gen_seq, tar_seq):
    gen_len_dist = compute_traj_len(gen_seq)
    tar_len_dist = compute_traj_len(tar_seq)
    if len(gen_len_dist) < len(tar_len_dist):
        gen_len_dist = np.array(gen_len_dist.tolist() + [0] * (len(tar_len_dist) - len(gen_len_dist)))
    elif len(tar_len_dist) < len(gen_len_dist):
        tar_len_dist = np.array(tar_len_dist.tolist() + [0] * (len(gen_len_dist) - len(tar_len_dist)))
    traj_len_jsd = distance.jensenshannon(gen_len_dist / np.sum(gen_len_dist), tar_len_dist / np.sum(tar_len_dist))
    return traj_len_jsd

def compute_hour(act_seq, n_time):
    act2hour = np.zeros((11, n_time)) # count of intervals of different activities
    for i in range(act_seq.shape[0]):
        curr_act = act_seq[i, 0]
        act2hour[curr_act, 0] = act2hour[curr_act, 0] + 1
        for j in range(1, act_seq.shape[1]):
            if act_seq[i, j] == curr_act:
                continue
            else:
                curr_act = act_seq[i, j]
                act2hour[curr_act, j] = act2hour[curr_act, j] + 1
    return act2hour

def macro_micro_hour_jsd(gen_seq, tar_seq, n_time):
    gen_act2hour = compute_hour(gen_seq, n_time)
    tar_act2hour = compute_hour(tar_seq, n_time)
    macro_hour_jsd = distance.jensenshannon(np.sum(gen_act2hour, 0) / np.sum(gen_act2hour), np.sum(tar_act2hour, 0) / np.sum(tar_act2hour))
    micro_hour_jsd = distance.jensenshannon(gen_act2hour.reshape(-1) / np.sum(gen_act2hour), tar_act2hour.reshape(-1) / np.sum(tar_act2hour))
    return macro_hour_jsd, micro_hour_jsd

def generated_tuple2seq(gen_tuples):
    gen_trajs = [[user_gen_tuple[0] for user_gen_tuple in user_gen_tuples] for user_gen_tuples in gen_tuples]
    return np.array(gen_trajs)

def evaluation(gen_seq, tar_seq, n_time):
    macro_int_jsd, micro_int_jsd = macro_micro_int_jsd(gen_seq, tar_seq, n_time)
    macro_hour_jsd, micro_hour_jsd = macro_micro_hour_jsd(gen_seq, tar_seq, n_time)
    results = {"accuracy": acc(gen_seq, tar_seq),
               "f1-score": f1(gen_seq, tar_seq),
               "edit_dist": edit_dist(gen_seq, tar_seq),
               "bleu_score": bleu_score(gen_seq, tar_seq),
               "data_jsd": dataset_jsd(gen_seq, tar_seq),
               "macro_int": macro_int_jsd,
               "micro_int": micro_int_jsd,
               "act_type": act_type_jsd(gen_seq, tar_seq),
               "uni_act_type": uni_act_type_jsd(gen_seq, tar_seq),
               "traj_len": traj_len_jsd(gen_seq, tar_seq),
               "macro_hour": macro_hour_jsd,
               "micro_hour": micro_hour_jsd}
    return results


## second part: eval log prob
def eval_log_prob(policy, test_trajs, batch_ind_feat, batch_ind_emp):
    log_prob_ls = []
    for i in range(batch_ind_feat.shape[0]):
        activity, time, dur, traj_len, dur_leave_home, dur_travel, ind_feat, ind_emp = env.reset(batch_ind_feat[i], batch_ind_emp[i])
        ind_feat_var = torch.tensor(ind_feat).float().unsqueeze(0).to(device)
        ind_emp_var = torch.tensor(ind_emp).long().unsqueeze(0).to(device)
        # get_log_prob(self, curr_activity, curr_tim, curr_dur, curr_traj_len, curr_dur_leave_home, curr_dur_travel, ind_feat, ind_emp, actions):
        seq_log_prob = 0
        for t in range(env.time_size):
            activity_var = torch.tensor(activity).long().unsqueeze(0).to(device)
            time_var = torch.tensor(time).long().unsqueeze(0).to(device)
            dur_var = torch.tensor(dur).long().unsqueeze(0).to(device)
            traj_len_var = torch.tensor(traj_len).long().unsqueeze(0).to(device)
            dur_leave_home_var = torch.tensor(dur_leave_home).long().unsqueeze(0).to(device)
            dur_travel_var = torch.tensor(dur_travel).long().unsqueeze(0).to(device)
            next_activity = test_trajs[i][t]
            next_activity_var = torch.tensor(next_activity).long().unsqueeze(0).to(device)
            with torch.no_grad():
                log_prob = policy.get_log_prob(activity_var, time_var, dur_var, traj_len_var, \
                                dur_leave_home_var, dur_travel_var, ind_feat_var, ind_emp_var, next_activity_var)
            seq_log_prob += log_prob.item()
            next_time, next_dur, next_traj_len, next_dur_leave_home, next_dur_travel, action, _, done = \
                env.step(activity, time, dur, traj_len, dur_leave_home, dur_travel, next_activity)
            if done:
                break
            activity, time, dur, traj_len, dur_leave_home, dur_travel = next_activity, next_time, next_dur, next_traj_len, next_dur_leave_home, next_dur_travel
        log_prob_ls.append(seq_log_prob)
    return np.mean(log_prob_ls)
