from experiments import sweep
from experiments.config import QUANTILE_GRID
from src.cec_problem import CEC2017Problem
from src.de_msr import DEMSR
from src.de_psr import DEPSR


def test_build_jobs_count_and_shape():
    jobs = sweep.build_jobs(["MSR-trial"], dims=[10], seeds=[1, 2], functions=[1, 3, 4])
    assert len(jobs) == len(QUANTILE_GRID) * 3 * 1 * 2
    name, hp, func_id, dim, seed = jobs[0]
    assert name == "MSR-trial" and dim == 10 and (func_id, seed) == (1, 1)


def test_grid_override_applies():
    jobs = sweep.build_jobs(["PSR-trial"], dims=[10], seeds=[1], functions=[1],
                            grid_override={"z_star": [-0.2, 0.0]})
    assert {hp for _, hp, *_ in jobs} == {-0.2, 0.0}


def test_build_constructs_msr_with_hyperparameter():
    p = CEC2017Problem(1, 10)
    de = sweep._build(sweep.CONFIGS["MSR-trial"], 0.65, p, 10, 1)
    assert isinstance(de, DEMSR)
    assert de.comparison_quantile == 0.65 and de.comparison_mode == "trial"
    assert de.target == p.optimum + sweep.TOL


def test_build_constructs_psr_with_hyperparameter():
    p = CEC2017Problem(1, 10)
    de = sweep._build(sweep.CONFIGS["PSR-population"], 0.3, p, 10, 1)
    assert isinstance(de, DEPSR)
    assert de.z_star == 0.3 and de.comparison_mode == "population"