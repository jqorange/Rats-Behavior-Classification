import pandas as pd


def modify_labels(df, syllable_name):
    # 获取Syllable列
    syllables = df[syllable_name].tolist()

    # 初始化变量
    current_label = syllables[0]
    count = 1

    # 遍历Syllable列
    for i in range(1, len(syllables)):
        if syllables[i] == current_label:
            count += 1
        else:
            if count < 6:
                for j in range(i - count, i):
                    syllables[j] = syllables[i - count - 1]
            current_label = syllables[i]
            count = 1

    # 处理最后一段
    if count < 3:
        for j in range(len(syllables) - count, len(syllables)):
            syllables[j] = syllables[len(syllables) - count - 1]

    # 更新DataFrame
    df[syllable_name] = syllables
    return df



# 使用示例
def modify_all_labels(session_names):
    syllable_names = ['walk', 'jump', 'aiming', 'scratch', 'rearing', 'stand_up', 'still', 'eating', 'local_search',
                      'turn_left', 'turn_right']
    for session in session_names:
        df = pd.read_csv(f'../Predictions_results/{session}_pred.csv')


        for name in syllable_names:
            df = modify_labels(df, name)

        # 重新排列Index列
        df['Index'] = range(len(df))

        df.to_csv(f'../Predictions_results/{session}_modified.csv', index=False)
session_names = ["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor","F5D10_outdoor", "F6D5_outdoor_2","F6D5_outdoor_1"]
modify_all_labels(session_names)