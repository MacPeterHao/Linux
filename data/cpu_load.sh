#!/bin/sh
# Most Hardcore text (Ingo Molnar - http://lkml.org/lkml/2005/6/22/347)
# For Heavy CPU Ratio
# ��ʼ��һ�����������洢��̨���̵�PID

pids=()

#����������
cleanup() {
	echo"�յ��źţ�����ֹͣ�����ɽű�����������..."
	#�������飬ɱ�����м�¼�ĺ�̨����
	for pid in "${pids[@]}";do
		kill "$pid" 2>/dev/null
	done
	echo "����������ֹͣ���ű��˳�"
	exit 0
}

#����trap������SIGINT�źţ�������cleanup����
trap cleanup SIGINT

#������̨���񣬲������ǵ�PID��ӵ�������
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
#�ȴ����к�̨�������
wait