import parmed as pmd

system = pmd.load_file('quick.gro', '1JNX.top')
system.save('1JNX.rst7')
