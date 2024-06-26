#!/bin/sh
# Most Hardcore text (Ingo Molnar - http://lkml.org/lkml/2005/6/22/347)
# For Heavy CPU Ratio
# 初始化一个空数组来存储后台进程的PID

pids=()

#定义清理函数
cleanup() {
	echo"收到信号，正在停止所有由脚本启动的任务..."
	#遍历数组，杀死所有记录的后台进程
	for pid in "${pids[@]}";do
		kill "$pid" 2>/dev/null
	done
	echo "所有任务已停止，脚本退出"
	exit 0
}

#设置trap来捕获SIGINT信号，并调用cleanup函数
trap cleanup SIGINT

#启动后台任务，并将它们的PID添加到数组中
while true; do /bin/dd if=/dev/zero of=bigfile bs=102400 count=1024; done &
pids+=($!)
while true; do /usr/bin/killall hackbench; sleep 5; done &
pids+=($!)
while true; do /sbin/hackbench 20; done &
pids+=($!)
# some source code(ltp-full-20090531) consists of sched_setschduler() with FIFO 99.
cd ltp-full-20090531; while true; do ./runalltests.sh -x 40; done &
pids+=($!)
chmod +x cputest.sh
#等待所有后台任务完成
wait