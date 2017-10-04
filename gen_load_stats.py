"""
This program generates GNU Octave script to load stats.csv
"""

import metrics

import sys


try:
    stats_file = sys.argv[1]
except IndexError:
    stats_file = 'stats.csv'


try:
    name_prefix = sys.argv[2]
except IndexError:
    name_prefix = ''

print("""% autogenereted by python gen_load_stats.py

{name_prefix}stats = csvread('{stats_file}');
# drop header
{name_prefix}stats = {name_prefix}stats(2:end, :);

c_clip = 1;
c_clipno = 2;
c_frame = 3;

clip1 = {name_prefix}stats(:, c_clipno) == 1;
clip2 = {name_prefix}stats(:, c_clipno) == 2;
clip3 = {name_prefix}stats(:, c_clipno) == 3;

""".format(name_prefix=name_prefix, stats_file=stats_file))

for num, name in enumerate(metrics.FRAME_METRICS):
    print('c_%s = %s;' % (name, num + 4))

print('');
    
for name in metrics.FRAME_METRICS:
    print('%s%s = %sstats(:, c_%s);'  % (name_prefix, name, name_prefix, name))
