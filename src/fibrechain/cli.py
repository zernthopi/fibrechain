"""Command line:  fibrechain run CONFIG [-o OUTDIR] [--steps a,b,...]"""
import argparse
from .pipeline import run


def main(argv=None):
    p = argparse.ArgumentParser(prog="fibrechain",
                                description="Lenslet-to-detector throughput of a fibre-fed "
                                            "spectrograph from the fibre angular distribution "
                                            "and a real ray trace of the spectrograph.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the whole computation for a configuration")
    r.add_argument("config")
    r.add_argument("-o", "--outdir", default="results")
    r.add_argument("--steps", default="fibre,maps,budget,uniform,hexagon,validate")
    a = p.parse_args(argv)
    if a.cmd == "run":
        run(a.config, a.outdir, tuple(s.strip() for s in a.steps.split(",")))


if __name__ == "__main__":
    main()
