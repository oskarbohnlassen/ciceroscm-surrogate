import sys
import os
import numpy as np
import torch
import copy

sys.path.insert(0,os.path.join(os.getcwd(), '../ciceroscm/', 'src')) ## Make ciceroscm importable
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURR_DIR)

from ciceroscm import CICEROSCM

class CICEROSCMEngine:
    """
    Reference engine that consumes full history 1900..t (G=40),
    calls CICEROSCM each step, and returns next-year temperature.
    """
    def __init__(
        self,
        historical_emissions,
        gaspam_data,
        conc_data,
        nat_ch4_data,
        nat_n2o_data,
        pamset_udm,
        pamset_emiconc,
        em_data_start=1900,
        em_data_policy=2015,
        udir=".",
        idtm=24,
        scenname="rl_scenario",
    ):
        # ----- store static inputs -----
        self.gaspam_data = gaspam_data
        self.conc_data = conc_data
        self.nat_ch4_data = nat_ch4_data
        self.nat_n2o_data = nat_n2o_data
        self.pamset_udm = pamset_udm
        self.pamset_emiconc = pamset_emiconc
        self.udir = udir
        self.idtm = int(idtm)
        self.scenname = str(scenname)

        self.em_df = historical_emissions
        self.current_year = em_data_policy
        self.T = 0.0  

        # ----- immutable scenario template -----
        self._scenario_template = {
            "gaspam_data": self.gaspam_data,
            "nyend": self.current_year,
            "nystart": int(em_data_start),
            "emstart": int(em_data_policy),
            "concentrations_data": self.conc_data,
            "nat_ch4_data": self.nat_ch4_data,
            "nat_n2o_data": self.nat_n2o_data,
            "emissions_data": self.em_df,  # will be replaced each step
            "udir": self.udir,
            "idtm": self.idtm,
            "scenname": self.scenname,
        }

    def _build_scenario(self):
        sc = copy.copy(self._scenario_template)
        sc["emissions_data"] = self.em_df
        sc["nyend"] = self.current_year
        return sc

    def step(self, E_t):
        """
        Append emissions E_t (shape (G,)) for year current_year+1,
        run CICEROSCM, and return (T_next, info).
        """
        next_year = int(self.current_year + 1)
        e = np.asarray(E_t, dtype=np.float32)

        # Append new year to emissions DF
        self.em_df = copy.copy(self.em_df)
        self.em_df.loc[next_year] = e
        self.current_year = next_year

        # Build scenario and run SCM (time just the engine)
        scenario = self._build_scenario()
        cscm = CICEROSCM(scenario)
        cscm._run({"results_as_dict": True},
                            pamset_udm=self.pamset_udm,
                            pamset_emiconc=self.pamset_emiconc)

        # Extract temperature for next_year
        T_next = cscm.results["dT_glob_air"][-1]

        self.T = float(T_next)
        return self.T


class CICERONetEngine:
    def __init__(
        self,
        historical_emissions,
        model,
        device="cuda:0",
        mu=None,
        std=None,
        autocast=True,
        use_half=True,
        window_size=65,
    ):

        self.window = int(window_size)
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.use_half = bool(use_half) and (self.device.type == "cuda")
        # Only enable autocast if we’re actually using half precision.
        self.autocast = bool(autocast) and self.use_half

        # Fast kernels (safe no-ops on CPU)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")    

        if historical_emissions.shape[0] < self.window:
            raise ValueError(
                "historical_emissions has fewer rows than window_size "
                f"({historical_emissions.shape[0]} < {self.window})"
            )

        # ---- keep exactly the last W rows (t-W..t-1) as the rolling buffer ----
        hist_tail = np.asarray(historical_emissions[-self.window :], dtype=np.float32)
        self.G = hist_tail.shape[1]
        self.buf = torch.from_numpy(hist_tail).to(self.device, non_blocking=True)  # (50, G)

        # Model input is (1, 51, G)
        dtype = torch.float16 if self.use_half else torch.float32
        self.x = torch.empty((1, self.window + 1, self.G), device=self.device, dtype=dtype)

        # Optional per-gas normalization stats (on device, dtype-matched)
        self.mu  = None if mu  is None else torch.as_tensor(mu,  device=self.device, dtype=self.x.dtype)
        self.std = None if std is None else torch.as_tensor(std, device=self.device, dtype=self.x.dtype)

        self.T = 0.5  # last predicted temperature (scalar float)

    @torch.inference_mode()
    def step(self, E_t):
        e = torch.as_tensor(E_t, device=self.device, dtype=self.buf.dtype)  # (G,)

        # Build (window+1, G) input: first the historical rows …
        self.x[0, : self.window].copy_(self.buf, non_blocking=True)
        # … then append current action-year emissions
        self.x[0, self.window].copy_(e, non_blocking=True)

        # Normalize inputs if stats provided: (x - mu)/std (broadcast over time)
        if (self.mu is not None) and (self.std is not None):
            self.x[0].sub_(self.mu).div_(self.std)

        # Forward pass
        if self.autocast:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model(self.x)
        else:
            out = self.model(self.x)

        T_next = float(out.squeeze().item())
        self.T = T_next

        # Update the rolling buffer for the next call: drop oldest, append E_t
        self.buf = torch.roll(self.buf, shifts=-1, dims=0)
        self.buf[-1].copy_(e, non_blocking=True)

        return self.T
