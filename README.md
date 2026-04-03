# skill-creator

Create new skills, improve existing skills, and measure skill performance. Use when users want to create a skill from scratch, update or optimize an existing skill, run evals to test a skill, or benchmark skill performance with variance analysis.

cd /mnt/d/skill-creator/brain-learn-main
bash run_wsl.sh scout-stop
bash run_wsl.sh scout-loop --manual-stop-only --poll-seconds 900 --clear-stop-file --count 6 --search-breadth explore --github-per-query 2 --github-query-limit 10 --github-readme-limit 6 --learn-zip-path /mnt/c/Users/OS/Downloads/worldquant-miner-master.zip --zip-seed-limit 10 --include-watchlist

