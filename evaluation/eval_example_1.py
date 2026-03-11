## file: eval
"""evaluation"""
import numpy as np
import editdistance
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu
from scipy.spatial import distance
from nltk.translate.bleu_score import SmoothingFunction
from datetime import datetime

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


# ============================================================================
# 家庭协同活动匹配率指标（Household Coordination Matching）
# ============================================================================


def compute_coordination_matching_rate(gen_household_trajectories, tar_household_trajectories, exclude_activities=None, time_tolerance_minutes=60):
    """
    计算协同活动匹配率 - 评估生成数据是否成功复现真实数据中的协同活动
    
    这个指标评估：真实数据中的协同活动，有多少被生成数据在相似时间段正确预测了
    不要求时间完全一致，允许一定的时间偏差（默认±60分钟）
    
    参数:
        gen_household_trajectories: 生成的家庭轨迹数据
        tar_household_trajectories: 真实的家庭轨迹数据
        exclude_activities: 要排除的活动类型列表（默认None表示包含所有活动）
        time_tolerance_minutes: 时间容错（分钟），默认60分钟
    
    返回:
        matching_rate: 匹配率 (0-1之间)
        details: {
            "total_target_coordinations": 真实数据中的总协同活动数,
            "matched_coordinations": 被成功预测的协同活动数,
            "matched_examples": 匹配成功的示例,
            "unmatched_examples": 未匹配的示例
        }
    """
    if exclude_activities is None:
        exclude_activities = []
    
    def time_to_minutes(time_str):
        """将时间字符串转换为分钟数"""
        if time_str == "24:00":
            return 1440
        parts = time_str.split(':')
        return int(parts[0]) * 60 + int(parts[1])
    
    def time_within_tolerance(time1, time2, tolerance):
        """判断两个时间点是否在容错范围内"""
        diff = abs(time_to_minutes(time1) - time_to_minutes(time2))
        return diff <= tolerance
    
    def activity_time_matches(start1, end1, start2, end2, tolerance):
        """
        判断两个活动的时间是否匹配（在容错范围内）
        要求：开始时间差 <= tolerance 且 结束时间差 <= tolerance
        """
        start_match = time_within_tolerance(start1, start2, tolerance)
        end_match = time_within_tolerance(end1, end2, tolerance)
        return start_match and end_match
    
    # 1. 收集真实数据中的所有协同活动
    target_coordinations = []
    for household_id, members in tar_household_trajectories.items():
        member_ids = list(members.keys())
        if len(member_ids) < 2:
            continue
        
        for i in range(len(member_ids)):
            for j in range(i + 1, len(member_ids)):
                member_i = member_ids[i]
                member_j = member_ids[j]
                
                # 找出真实数据中时间完全相同且活动相同的活动对
                for activity_i in members[member_i]:
                    activity_type = activity_i.get('activity', 'unknown')
                    
                    # 排除指定活动
                    if activity_type in exclude_activities:
                        continue
                    
                    start_i = activity_i.get('start_time', '00:00')
                    end_i = activity_i.get('end_time', '24:00')
                    
                    for activity_j in members[member_j]:
                        if activity_j.get('activity') != activity_type:
                            continue
                        
                        start_j = activity_j.get('start_time', '00:00')
                        end_j = activity_j.get('end_time', '24:00')
                        
                        # 时间完全相同且活动相同 = 真实协同活动
                        if start_i == start_j and end_i == end_j:
                            target_coordinations.append({
                                "household_id": household_id,
                                "member_i": member_i,
                                "member_j": member_j,
                                "activity": activity_type,
                                "start_time": start_i,
                                "end_time": end_i,
                                "target_trajectory_i": members[member_i],
                                "target_trajectory_j": members[member_j]
                            })
    
    # 2. 检查这些协同活动在生成数据中是否也存在
    matched_coordinations = []
    unmatched_coordinations = []
    
    for target_coord in target_coordinations:
        household_id = target_coord["household_id"]
        member_i = target_coord["member_i"]
        member_j = target_coord["member_j"]
        activity_type = target_coord["activity"]
        start_time = target_coord["start_time"]
        end_time = target_coord["end_time"]
        
        # 检查生成数据中是否有该家庭
        if household_id not in gen_household_trajectories:
            unmatched_coordinations.append(target_coord)
            continue
        
        gen_members = gen_household_trajectories[household_id]
        
        # 检查成员是否存在
        if member_i not in gen_members or member_j not in gen_members:
            unmatched_coordinations.append(target_coord)
            continue
        
        # 检查生成数据中这两个成员在相似时间段是否做相同活动（允许时间容错）
        found_match = False
        for gen_activity_i in gen_members[member_i]:
            if gen_activity_i.get('activity') != activity_type:
                continue
            
            # 检查时间是否在容错范围内（开始和结束时间都要在容错范围内）
            gen_start_i = gen_activity_i.get('start_time', '00:00')
            gen_end_i = gen_activity_i.get('end_time', '24:00')
            
            if not activity_time_matches(start_time, end_time, gen_start_i, gen_end_i, time_tolerance_minutes):
                continue
            
            # member_i时间匹配，检查member_j
            for gen_activity_j in gen_members[member_j]:
                if gen_activity_j.get('activity') != activity_type:
                    continue
                
                gen_start_j = gen_activity_j.get('start_time', '00:00')
                gen_end_j = gen_activity_j.get('end_time', '24:00')
                
                if not activity_time_matches(start_time, end_time, gen_start_j, gen_end_j, time_tolerance_minutes):
                    continue
                
                # 检查两个成员的生成活动时间是否也相同（协同）
                if gen_start_i == gen_start_j and gen_end_i == gen_end_j:
                    # 找到匹配的协同活动
                    found_match = True
                    matched_coordinations.append({
                        **target_coord,
                        "gen_start_time": gen_start_i,
                        "gen_end_time": gen_end_i,
                        "time_difference_start": abs(time_to_minutes(start_time) - time_to_minutes(gen_start_i)),
                        "time_difference_end": abs(time_to_minutes(end_time) - time_to_minutes(gen_end_i))
                    })
                    break
            
            if found_match:
                break
        
        if not found_match:
            # 收集该家庭成员在生成数据中的实际轨迹
            gen_trajectory_i = []
            gen_trajectory_j = []
            
            if household_id in gen_household_trajectories:
                gen_members_hh = gen_household_trajectories[household_id]
                if member_i in gen_members_hh:
                    gen_trajectory_i = gen_members_hh[member_i]
                if member_j in gen_members_hh:
                    gen_trajectory_j = gen_members_hh[member_j]
            
            unmatched_coordinations.append({
                **target_coord,
                "reason": "no_matching_coordination_in_gen",
                "gen_trajectory_i": gen_trajectory_i,
                "gen_trajectory_j": gen_trajectory_j
            })
    
    # 3. 计算匹配率
    total = len(target_coordinations)
    matched = len(matched_coordinations)
    
    # 如果没有真实协同活动，返回 NaN 表示无法评估
    # 如果有协同活动但都没匹配，返回 0.0
    if total == 0:
        matching_rate = np.nan
    else:
        matching_rate = matched / total
    
    # 计算统计信息
    activity_type_stats = {}
    for coord in target_coordinations:
        act = coord['activity']
        if act not in activity_type_stats:
            activity_type_stats[act] = {'total': 0, 'matched': 0}
        activity_type_stats[act]['total'] += 1
    
    for coord in matched_coordinations:
        act = coord['activity']
        if act in activity_type_stats:
            activity_type_stats[act]['matched'] += 1
    
    return matching_rate, {
        "total_target_coordinations": total,
        "matched_coordinations": matched,
        "matched_examples": matched_coordinations[:10],  # 增加到10个示例
        "unmatched_examples": unmatched_coordinations[:10],
        "activity_type_stats": activity_type_stats,
        "all_matched": matched_coordinations,
        "all_unmatched": unmatched_coordinations
    }


def generate_coordination_report(matching_details_all, matching_details_exclude_work, output_path):
    """
    生成家庭协同活动匹配的详细报告文档
    
    参数:
        matching_details_all: 全部活动的匹配详情
        matching_details_exclude_work: 排除work活动的匹配详情
        output_path: 输出文档路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("家庭协同活动匹配评估报告\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # ===== 总体匹配率摘要 =====
        total_all = matching_details_all['total_target_coordinations']
        matched_all = matching_details_all['matched_coordinations']
        
        total_exclude = matching_details_exclude_work['total_target_coordinations']
        matched_exclude = matching_details_exclude_work['matched_coordinations']
        
        f.write("【总体匹配率】\n")
        f.write("="*80 + "\n")
        
        # 格式化匹配率显示
        if total_all > 0:
            rate_all_str = f"{matched_all/total_all:>6.2%}"
        else:
            rate_all_str = "  N/A  "
        
        if total_exclude > 0:
            rate_exclude_str = f"{matched_exclude/total_exclude:>6.2%}"
        else:
            rate_exclude_str = "  N/A  "
        
        f.write(f"全部活动协同匹配率:        {rate_all_str}  ({matched_all}/{total_all})\n")
        f.write(f"排除work活动协同匹配率:   {rate_exclude_str}  ({matched_exclude}/{total_exclude})\n")
        
        if total_exclude == 0:
            f.write("\n注意: 真实数据中没有非work类型的协同活动，无法评估匹配率\n")
        
        f.write("="*80 + "\n\n")
        
        f.write("评估说明:\n")
        f.write("-" * 80 + "\n")
        f.write("1. 目标: 评估生成数据是否成功复现真实数据中的家庭协同活动\n")
        f.write("2. 时间容错: ±60分钟（生成活动与真实活动比较时允许误差）\n")
        f.write("3. 协同判定: 生成数据内部，家庭成员的活动时间必须完全相同才算协同\n")
        f.write("4. 部分协同: 不要求所有家庭成员一致，任意2个或更多成员同时做同一活动即算协同\n")
        f.write("   例如: 3口之家，爸妈一起shopping（19:00-20:00），孩子在home，\n")
        f.write("         爸妈的shopping仍算作一个协同活动\n")
        f.write("\n")
        
        # ===== 全部活动匹配情况 =====
        f.write("="*80 + "\n")
        f.write("【详细分析 1】全部活动匹配情况\n")
        f.write("="*80 + "\n\n")
        
        if total_all > 0:
            rate_all = matched_all / total_all
            f.write(f"匹配率: {rate_all:.2%} ({matched_all}/{total_all})\n\n")
        else:
            f.write(f"匹配率: N/A (0/0) - 真实数据中没有协同活动\n\n")
        
        # 按家庭统计
        household_stats = {}
        for coord in matching_details_all['all_matched'] + matching_details_all['all_unmatched']:
            hh_id = coord['household_id']
            if hh_id not in household_stats:
                household_stats[hh_id] = {'total': 0, 'matched': 0}
            household_stats[hh_id]['total'] += 1
        
        for coord in matching_details_all['all_matched']:
            hh_id = coord['household_id']
            household_stats[hh_id]['matched'] += 1
        
        if household_stats:
            f.write("按家庭统计:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'家庭ID':<15} {'总协同数':>10} {'成功预测':>10} {'匹配率':>10}\n")
            f.write("-" * 80 + "\n")
            
            for hh_id, stats in sorted(household_stats.items()):
                total = stats['total']
                matched = stats['matched']
                rate = matched / total if total > 0 else 0
                f.write(f"{hh_id:<15} {total:>10} {matched:>10} {rate:>9.1%}\n")
            f.write("\n")
        
        # 按活动类型统计
        if 'activity_type_stats' in matching_details_all:
            f.write("按活动类型统计:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'活动类型':<15} {'总协同数':>10} {'成功预测':>10} {'匹配率':>10}\n")
            f.write("-" * 80 + "\n")
            
            for activity, stats in sorted(matching_details_all['activity_type_stats'].items()):
                total = stats['total']
                matched = stats['matched']
                rate = matched / total if total > 0 else 0
                f.write(f"{activity:<15} {total:>10} {matched:>10} {rate:>9.1%}\n")
            f.write("\n")
        
        # 匹配成功的示例
        if matching_details_all['all_matched']:
            f.write(f"\n✓ 匹配成功的协同活动 (共 {len(matching_details_all['all_matched'])} 个):\n")
            f.write("-" * 80 + "\n")
            for i, coord in enumerate(matching_details_all['all_matched'], 1):
                f.write(f"{i}. 家庭ID: {coord['household_id']}\n")
                f.write(f"   成员: {coord['member_i']} 和 {coord['member_j']}\n")
                f.write(f"   活动: {coord['activity']}\n")
                f.write(f"   真实时间: {coord['start_time']} - {coord['end_time']}\n")
                if 'gen_start_time' in coord:
                    f.write(f"   生成时间: {coord['gen_start_time']} - {coord['gen_end_time']}\n")
                    start_diff = coord.get('time_difference_start', 0)
                    end_diff = coord.get('time_difference_end', 0)
                    f.write(f"   时间偏差: 开始 {start_diff}分钟, 结束 {end_diff}分钟\n")
                f.write("\n")
        
        # 未匹配的示例
        if matching_details_all['all_unmatched']:
            f.write(f"\n✗ 未匹配的协同活动 (共 {len(matching_details_all['all_unmatched'])} 个):\n")
            f.write("-" * 80 + "\n")
            for i, coord in enumerate(matching_details_all['all_unmatched'], 1):
                f.write(f"\n{i}. 家庭ID: {coord['household_id']}\n")
                f.write(f"   成员: {coord['member_i']} 和 {coord['member_j']}\n")
                f.write(f"   协同活动: {coord['activity']}\n")
                f.write(f"   协同时间: {coord['start_time']} - {coord['end_time']}\n")
                f.write(f"   原因: 生成数据中未找到对应的协同活动\n\n")
                
                # 显示真实轨迹
                f.write(f"   【真实轨迹】\n")
                if 'target_trajectory_i' in coord:
                    f.write(f"   {coord['member_i']}:\n")
                    for act in coord['target_trajectory_i']:
                        f.write(f"      {act.get('start_time', '')} - {act.get('end_time', '')}: {act.get('activity', '')}\n")
                    
                    f.write(f"   {coord['member_j']}:\n")
                    for act in coord['target_trajectory_j']:
                        f.write(f"      {act.get('start_time', '')} - {act.get('end_time', '')}: {act.get('activity', '')}\n")
                
                # 显示生成轨迹
                f.write(f"\n   【生成轨迹】\n")
                if 'gen_trajectory_i' in coord and coord['gen_trajectory_i']:
                    f.write(f"   {coord['member_i']}:\n")
                    for act in coord['gen_trajectory_i']:
                        f.write(f"      {act.get('start_time', '')} - {act.get('end_time', '')}: {act.get('activity', '')}\n")
                else:
                    f.write(f"   {coord['member_i']}: (无生成轨迹)\n")
                
                if 'gen_trajectory_j' in coord and coord['gen_trajectory_j']:
                    f.write(f"   {coord['member_j']}:\n")
                    for act in coord['gen_trajectory_j']:
                        f.write(f"      {act.get('start_time', '')} - {act.get('end_time', '')}: {act.get('activity', '')}\n")
                else:
                    f.write(f"   {coord['member_j']}: (无生成轨迹)\n")
                
                f.write("\n" + "-" * 80 + "\n")
        
        # ===== 排除work活动匹配情况 =====
        f.write("\n" + "="*80 + "\n")
        f.write("【详细分析 2】排除work活动匹配情况（专注于非工作协同）\n")
        f.write("="*80 + "\n\n")
        
        if total_exclude > 0:
            rate_exclude = matched_exclude / total_exclude
            f.write(f"匹配率: {rate_exclude:.2%} ({matched_exclude}/{total_exclude})\n\n")
        else:
            f.write(f"匹配率: N/A (0/0) - 真实数据中没有非work类型的协同活动\n\n")
        
        # 按家庭统计
        household_stats_exclude = {}
        for coord in matching_details_exclude_work['all_matched'] + matching_details_exclude_work['all_unmatched']:
            hh_id = coord['household_id']
            if hh_id not in household_stats_exclude:
                household_stats_exclude[hh_id] = {'total': 0, 'matched': 0}
            household_stats_exclude[hh_id]['total'] += 1
        
        for coord in matching_details_exclude_work['all_matched']:
            hh_id = coord['household_id']
            household_stats_exclude[hh_id]['matched'] += 1
        
        if household_stats_exclude:
            f.write("按家庭统计:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'家庭ID':<15} {'总协同数':>10} {'成功预测':>10} {'匹配率':>10}\n")
            f.write("-" * 80 + "\n")
            
            for hh_id, stats in sorted(household_stats_exclude.items()):
                total = stats['total']
                matched = stats['matched']
                rate = matched / total if total > 0 else 0
                f.write(f"{hh_id:<15} {total:>10} {matched:>10} {rate:>9.1%}\n")
            f.write("\n")
        
        # 按活动类型统计
        if 'activity_type_stats' in matching_details_exclude_work:
            f.write("按活动类型统计:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'活动类型':<15} {'总协同数':>10} {'成功预测':>10} {'匹配率':>10}\n")
            f.write("-" * 80 + "\n")
            
            for activity, stats in sorted(matching_details_exclude_work['activity_type_stats'].items()):
                total = stats['total']
                matched = stats['matched']
                rate = matched / total if total > 0 else 0
                f.write(f"{activity:<15} {total:>10} {matched:>10} {rate:>9.1%}\n")
            f.write("\n")
        
        # 匹配成功的示例
        if matching_details_exclude_work['all_matched']:
            f.write(f"\n✓ 匹配成功的协同活动 (共 {len(matching_details_exclude_work['all_matched'])} 个):\n")
            f.write("-" * 80 + "\n")
            for i, coord in enumerate(matching_details_exclude_work['all_matched'], 1):
                f.write(f"{i}. 家庭ID: {coord['household_id']}\n")
                f.write(f"   成员: {coord['member_i']} 和 {coord['member_j']}\n")
                f.write(f"   活动: {coord['activity']}\n")
                f.write(f"   真实时间: {coord['start_time']} - {coord['end_time']}\n")
                if 'gen_start_time' in coord:
                    f.write(f"   生成时间: {coord['gen_start_time']} - {coord['gen_end_time']}\n")
                    start_diff = coord.get('time_difference_start', 0)
                    end_diff = coord.get('time_difference_end', 0)
                    f.write(f"   时间偏差: 开始 {start_diff}分钟, 结束 {end_diff}分钟\n")
                f.write("\n")
        
        # 未匹配的示例
        if matching_details_exclude_work['all_unmatched']:
            f.write(f"\n✗ 未匹配的协同活动 (共 {len(matching_details_exclude_work['all_unmatched'])} 个):\n")
            f.write("-" * 80 + "\n")
            for i, coord in enumerate(matching_details_exclude_work['all_unmatched'], 1):
                f.write(f"\n{i}. 家庭ID: {coord['household_id']}\n")
                f.write(f"   成员: {coord['member_i']} 和 {coord['member_j']}\n")
                f.write(f"   协同活动: {coord['activity']}\n")
                f.write(f"   协同时间: {coord['start_time']} - {coord['end_time']}\n")
                f.write(f"   原因: 生成数据中未找到对应的协同活动\n\n")
                
                # 显示真实轨迹
                f.write(f"   【真实轨迹】\n")
                if 'target_trajectory_i' in coord:
                    f.write(f"   {coord['member_i']}:\n")
                    for act in coord['target_trajectory_i']:
                        f.write(f"      {act.get('start_time', '')} - {act.get('end_time', '')}: {act.get('activity', '')}\n")
                    
                    f.write(f"   {coord['member_j']}:\n")
                    for act in coord['target_trajectory_j']:
                        f.write(f"      {act.get('start_time', '')} - {act.get('end_time', '')}: {act.get('activity', '')}\n")
                
                # 显示生成轨迹
                f.write(f"\n   【生成轨迹】\n")
                if 'gen_trajectory_i' in coord and coord['gen_trajectory_i']:
                    f.write(f"   {coord['member_i']}:\n")
                    for act in coord['gen_trajectory_i']:
                        f.write(f"      {act.get('start_time', '')} - {act.get('end_time', '')}: {act.get('activity', '')}\n")
                else:
                    f.write(f"   {coord['member_i']}: (无生成轨迹)\n")
                
                if 'gen_trajectory_j' in coord and coord['gen_trajectory_j']:
                    f.write(f"   {coord['member_j']}:\n")
                    for act in coord['gen_trajectory_j']:
                        f.write(f"      {act.get('start_time', '')} - {act.get('end_time', '')}: {act.get('activity', '')}\n")
                else:
                    f.write(f"   {coord['member_j']}: (无生成轨迹)\n")
                
                f.write("\n" + "-" * 80 + "\n")
        
        f.write("="*80 + "\n")
        f.write("报告结束\n")
        f.write("="*80 + "\n")
    
    print(f"✓ 协同活动匹配报告已保存: {output_path}")


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
