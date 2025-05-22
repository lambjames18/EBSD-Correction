import Inputs
import Outputs

path = (
    "/Users/jameslamb/Documents/research/data/CoNi90-thin/ANG/slice_0114_Rescan_Mod.ang"
)

data, _ = Inputs.read_ang(path)
Outputs.write_ang_file(path, path.replace(".ang", "_out.ang"), data)
