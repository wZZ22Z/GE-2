import time
import argparse

def main():
    # Open the file
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, required=True)
    args = parser.parse_args()

    all_time = []
    with open(args.log_file, "r") as f:
        for line in f:
            if "Epoch Runtime" in line:
                # print(line.strip().split(" ")[-1][:-2])
                all_time.append(float(line.strip().split(" ")[-1][:-2]))
                print(f"Epoch {all_time[-1]}")
    print("Total time: ", sum(all_time) / len(all_time) / 1000)

if __name__ == "__main__":
    main()
