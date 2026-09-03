reinitialize

# LOAD
load 1JNX_PRO_1.pdb, prot
load_traj 1JNX_PRO1-1_whole.dcd, prot

# STYLE
hide everything
show cartoon, prot
bg_color pink

set antialias, 1
set ray_shadows, off
set cartoon_fancy_helices, 1

spectrum b, blue_white_red, prot

# CREATE REFERENCE (STATE 1)
create ref, prot
frame 1
remove ref and not (name CA)

# ALIGN ALL FRAMES PROPERLY
python
from pymol import cmd

obj = "prot"
ref = "ref"
n_states = cmd.count_states(obj)

for i in range(1, n_states + 1):
    cmd.frame(i)
    cmd.align(f"{obj} and name CA", f"{ref} and name CA")
    
    if i % 1 == 0:
        cmd.png(f"frames/frame_{i:04d}.png",
                width=1280, height=720, dpi=72, ray=0)

python end

quit
