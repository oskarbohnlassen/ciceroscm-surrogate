from pathlib import Path

import ciceroscm.input_handler as input_handler


def load_cicero_inputs(cicero_config):
    test_data_dir = Path(cicero_config["test_data_dir"])
    conc_first = cicero_config["conc_data_first"]
    conc_last = cicero_config["conc_data_last"]
    em_data_start = cicero_config["em_data_start"]
    em_data_policy = cicero_config["em_data_policy"]
    em_data_end = cicero_config["em_data_end"]

    gaspam_data = input_handler.read_components(test_data_dir / "gases_v1RCMIP.txt")
    conc_data = input_handler.read_inputfile(
        test_data_dir / "ssp245_conc_RCMIP.txt",
        True,
        conc_first,
        conc_last,
    )

    ih = input_handler.InputHandler(
        {"nyend": em_data_end, "nystart": em_data_start, "emstart": em_data_policy}
    )

    em_data = ih.read_emissions(test_data_dir / "ssp245_em_RCMIP.txt")
    nat_ch4_data = input_handler.read_natural_emissions(
        test_data_dir / "natemis_ch4.txt", "CH4"
    )
    nat_n2o_data = input_handler.read_natural_emissions(
        test_data_dir / "natemis_n2o.txt", "N2O"
    )

    return gaspam_data, conc_data, em_data, nat_ch4_data, nat_n2o_data
