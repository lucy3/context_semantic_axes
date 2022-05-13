import string

for letter in string.ascii_lowercase[:17]: 
    with open('./jobfiles/coref_' + letter + '.job', 'w') as outfile:
        outfile.write('#!/bin/bash\n')
        outfile.write('# Job name:\n')
        outfile.write('#SBATCH --job-name=coref_' + letter + '\n')
        outfile.write('# Partition:\n')
        outfile.write('#SBATCH --partition=savio\n')
        outfile.write('#\n')
        outfile.write('# Wall clock limit:\n')
        outfile.write('#SBATCH --time=3-00:00:00\n')
        outfile.write('#SBATCH --account=fc_dbamman\n')
        outfile.write('#SBATCH --nodes=3\n')
        outfile.write('#SBATCH --ntasks=1\n')
        outfile.write('#SBATCH --cpus-per-task=6\n')
        outfile.write('#SBATCH --mail-user=lucy3_li@berkeley.edu\n')
        outfile.write('#SBATCH --mail-type=all\n')


        outfile.write('source /global/scratch/users/lucy3_li/anaconda3/bin/activate /global/scratch/users/lucy3_li/anaconda3/envs/manosphere\n')

        outfile.write('time awk -F "," \'{print $1}\' ../../logs/data_splits/xa' + letter + ' | parallel --joblog ../../logs/coref_tasklogs/task_' + letter + '.log --resume \'python ../coref_reddit.py {}\' &\n')

        outfile.write('wait\n')

        outfile.write('source /global/scratch/users/lucy3_li/anaconda3/bin/deactivate\n')


for letter in string.ascii_lowercase[:17]: 
    with open('./jobfiles/coref_' + letter + '_control.job', 'w') as outfile:
        outfile.write('#!/bin/bash\n')
        outfile.write('# Job name:\n')
        outfile.write('#SBATCH --job-name=coref_' + letter + '_control\n')
        outfile.write('# Partition:\n')
        outfile.write('#SBATCH --partition=savio\n')
        outfile.write('#\n')
        outfile.write('# Wall clock limit:\n')
        outfile.write('#SBATCH --time=3-00:00:00\n')
        outfile.write('#SBATCH --account=fc_dbamman\n')
        outfile.write('#SBATCH --nodes=3\n')
        outfile.write('#SBATCH --ntasks=1\n')
        outfile.write('#SBATCH --cpus-per-task=6\n')
        outfile.write('#SBATCH --mail-user=lucy3_li@berkeley.edu\n')
        outfile.write('#SBATCH --mail-type=all\n')


        outfile.write('source /global/scratch/users/lucy3_li/anaconda3/bin/activate /global/scratch/users/lucy3_li/anaconda3/envs/manosphere\n')

        outfile.write('time awk -F "," \'{print $1}\' ../../logs/data_splits/xa' + letter + ' | parallel --joblog ../../logs/coref_tasklogs/task_' + letter + '_control.log --resume \'python ../coref_reddit_control.py {}\' &\n')

        outfile.write('wait\n')

        outfile.write('source /global/scratch/users/lucy3_li/anaconda3/bin/deactivate\n')

for letter in string.ascii_lowercase[:7]: 
    with open('./jobfiles/coref_forum_' + letter + '.job', 'w') as outfile:
        outfile.write('#!/bin/bash\n')
        outfile.write('# Job name:\n')
        outfile.write('#SBATCH --job-name=coref_forum_' + letter + '\n')
        outfile.write('# Partition:\n')
        outfile.write('#SBATCH --partition=savio\n')
        outfile.write('#\n')
        outfile.write('# Wall clock limit:\n')
        outfile.write('#SBATCH --time=3-00:00:00\n')
        outfile.write('#SBATCH --account=fc_dbamman\n')
        outfile.write('#SBATCH --nodes=3\n')
        outfile.write('#SBATCH --ntasks=1\n')
        outfile.write('#SBATCH --cpus-per-task=6\n')
        outfile.write('#SBATCH --mail-user=lucy3_li@berkeley.edu\n')
        outfile.write('#SBATCH --mail-type=all\n')


        outfile.write('source /global/scratch/users/lucy3_li/anaconda3/bin/activate /global/scratch/users/lucy3_li/anaconda3/envs/manosphere\n')

        outfile.write('time awk -F "," \'{print $1}\' ../../logs/data_splits/ya' + letter + ' | parallel --joblog ../../logs/coref_tasklogs/task_forums_' + letter + '.log --resume \'python ../coref_forums.py {}\' &\n')

        outfile.write('wait\n')

        outfile.write('source /global/scratch/users/lucy3_li/anaconda3/bin/deactivate\n')


