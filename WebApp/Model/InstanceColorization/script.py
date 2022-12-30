import os
import argparse
import glob
# parser = argparse.ArgumentParser(description='box number')
# parser.add_argument('--box', '-b', type=int, default=8, help='box')
# args = parser.parse_args()

# os.system(f"python test_fusion.py --name test_fusion --sample_p 1.0 --model fusion --fineSize 256 --test_img_dir example --results_img_dir results --box_num={args.box}")


# for box in range(1,9):  
#     os.system(f"python test_fusion.py --name test_fusion --sample_p 1.0 --model fusion --fineSize 256 --test_img_dir example --results_img_dir results --box_num={box}")

a = glob.glob("/home/saebyeol/colorization/InstColorization/example/test/*_bbox/")
for i in a:
    os.system(f"python test_fusion.py --name test_fusion --sample_p 1.0 --model fusion --fineSize 256 --test_img_dir {i[:-6]} --results_img_dir results --exp {i[-16:-10]}")
# "python test_fusion.py --name test_fusion --sample_p 1.0 --model fusion --fineSize 256 --test_img_dir ./example/test_08_15 --results_img_dir results --exp 08_15_test}