import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


run_id = "20250805_152136"

em_data_policy = 2015
WINDOW   = 50
data_output_dir = "/home/obola/repositories/cicero-scm-surrogate/data/"
data_output_dir_run = os.path.join(data_output_dir, run_id)


def load_simulation_data():
    data_output_dir_run_raw = os.path.join(data_output_dir_run, "raw")

    # Load pickled results and scenarios
    with open(os.path.join(data_output_dir_run_raw, "results.pkl"), "rb") as f:
        results = pickle.load(f)

    with open(os.path.join(data_output_dir_run_raw, "scenarios.pkl"), "rb") as f:
        scenarios = pickle.load(f)

    return results, scenarios


def format_results(results, scenarios):
    GASES    = scenarios[0]["emissions_data"].columns.tolist()

    temp_tbl = (
        results[ results["variable"] == "Surface Air Temperature Change" ]
        .set_index("scenario")          
        .filter(regex=r"^\d")           # keep only year columns
        .astype(float))
    
    # Ensure year columns are ints (not strings)
    temp_tbl.columns = temp_tbl.columns.astype(int)

    scen_trainval, scen_test = train_test_split(
        scenarios, test_size=0.15, random_state=0, shuffle=True
    )

    val_prop_within_trainval = 0.15 / 0.85
    scen_train, scen_val = train_test_split(
        scen_trainval, test_size=val_prop_within_trainval, random_state=0, shuffle=True
    )

    def generate_machine_learning_data(scenario_list):
        X_list, y_list = [], []
        for scen in scenario_list:
            name  = scen["scenname"]
            em_df = scen["emissions_data"]      
            years = em_df.index     

            T_air = temp_tbl.loc[name, years].to_numpy()

            for t_idx in range(WINDOW, len(years)-1):
                t_target = years[t_idx+1]       
                if t_target < em_data_policy:
                    continue                    

                hist = em_df.loc[t_target-WINDOW : t_target-1, GASES].to_numpy()
                next_em  = em_df.loc[t_target, GASES].to_numpy()
                X_sample = np.vstack([hist, next_em[None, :]]).astype("float32")  # (51,G)
                y      = np.float32(T_air[t_idx+1])                             # scalar

                X_list.append(X_sample)
                y_list.append(y)
        
        # convert to arrays
        X = np.stack(X_list)
        y = np.stack(y_list)
        return X, y
    
    # ---- Create datasets for each split ----
    X_train, y_train = generate_machine_learning_data(scen_train)
    X_val, y_val     = generate_machine_learning_data(scen_val)
    X_test, y_test   = generate_machine_learning_data(scen_test)

    return (X_train, y_train, X_val, y_val, X_test, y_test, scen_train, scen_val, scen_test)


def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test,
                        train_scenarios, val_scenarios, test_scenarios):
    data_output_dir_run_processed = os.path.join(data_output_dir_run, "processed")
    os.makedirs(data_output_dir_run_processed, exist_ok=True)

    np.save(os.path.join(data_output_dir_run_processed, "X_train.npy"), X_train)
    np.save(os.path.join(data_output_dir_run_processed, "y_train.npy"), y_train)
    np.save(os.path.join(data_output_dir_run_processed, "X_val.npy"), X_val)
    np.save(os.path.join(data_output_dir_run_processed, "y_val.npy"), y_val)
    np.save(os.path.join(data_output_dir_run_processed, "X_test.npy"), X_test)
    np.save(os.path.join(data_output_dir_run_processed, "y_test.npy"), y_test)

    # Save scenario lists as pickles to preserve dict structure
    with open(os.path.join(data_output_dir_run_processed, "train_scenarios.pkl"), "wb") as f:
        pickle.dump(train_scenarios, f)
    with open(os.path.join(data_output_dir_run_processed, "val_scenarios.pkl"), "wb") as f:
        pickle.dump(val_scenarios, f)
    with open(os.path.join(data_output_dir_run_processed, "test_scenarios.pkl"), "wb") as f:
        pickle.dump(test_scenarios, f)


def main():
    results, scenarios = load_simulation_data()
    (X_train, y_train, X_val, y_val, X_test, y_test, train_scenarios, val_scenarios, test_scenarios) = format_results(results, scenarios)
    save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, train_scenarios, val_scenarios, test_scenarios)

if __name__ == "__main__":
    main()
