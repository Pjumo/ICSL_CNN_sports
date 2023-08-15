import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 시작 디렉토리
start_dir = './../pkl_file'

# 이미지 파일을 저장할 디렉토리
img_dir = 'results'
os.makedirs(img_dir, exist_ok=True)  # 디렉토리가 없으면 생성

# 시작 디렉토리와 그 하위의 모든 디렉토리에 대해 처리
for dirpath, dirnames, _ in os.walk(start_dir):
    for dirname in dirnames:
        if dirname == 'valleyball':
            sports = os.path.join(dirpath, dirname)
            for filepath, _, filenames in os.walk(sports):
                cnt = 0
                for filename in filenames:
                    if cnt == 100:
                        break
                    cnt += 1
                    if filename.endswith('.pkl'):
                        # PKL 파일을 데이터프레임으로 읽기
                        df = pd.read_pickle(os.path.join(filepath, filename))
                        xyz_list = df['C']

                        # 전체 프레임의 수
                        num_frames = len(xyz_list)

                        fig = plt.figure(figsize=(8, 8))
                        ax = fig.add_subplot(111, projection='3d')


                        # 애니메이션 단계 함수: 각 프레임에서 수행할 작업
                        def animate(i):
                            for xyz in xyz_list[i]:
                                if xyz == '':
                                    return
                            data_np = np.array(xyz_list[i], dtype=float).reshape(-1, 3)
                            xyz_data = pd.DataFrame(data_np)
                            xyz_data.columns = ['x', 'y', 'z']
                            ax.scatter(xyz_data['x'], xyz_data['y'], xyz_data['z'], c=xyz_data['y'])
                            ax.set_xlabel('X')
                            ax.set_ylabel('Y')
                            ax.set_zlabel('Z')
                            ax.set_xlim([-2.5, 2.5])
                            ax.set_ylim([1, 4.5])
                            ax.set_zlim([-2, 0])
                            ax.view_init(0, -90)


                        # 모든 프레임에 대해 animate 함수 실행
                        for i in range(num_frames):
                            animate(i)

                        # CSV 파일명에서 확장자를 제거하고, 이를 이미지 파일명으로 사용
                        img_filename = f'{os.path.splitext(filename)[0]}.png'
                        img_dir_final = os.path.join(img_dir, dirname)
                        os.makedirs(img_dir_final, exist_ok=True)
                        img_path = os.path.join(img_dir_final, img_filename)  # 저장 경로 생성
                        plt.savefig(img_path)

                        plt.close()
