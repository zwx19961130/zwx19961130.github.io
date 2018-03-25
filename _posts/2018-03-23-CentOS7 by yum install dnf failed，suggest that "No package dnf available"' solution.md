---
layout: post
title: "CentOS7 by yum install dnf failed，suggest that "No package dnf available"' solution"
date: 2018-03-23
---


I want to install docker on CentOS 7 , and the command is 
	
	dnf -y install docker
	systemctl start docker 
	systemctl enable docker 
	docker pull fedora 
	docker run fedora /bin/echo "Welcome to the Docker World" 
When I type in the first command, I didn't succeed. The dnf could not be installed by
	
	yum install dnf
	
It says that there's no available dnf package. 	

The solution is as follows:

	wget http://springdale.math.ias.edu/data/puias/unsupported/7/x86_64/dnf-conf-0.6.4-2.sdl7.noarch.rpm
	wget http://springdale.math.ias.edu/data/puias/unsupported/7/x86_64//dnf-0.6.4-2.sdl7.noarch.rpm
	wget http://springdale.math.ias.edu/data/puias/unsupported/7/x86_64/python-dnf-0.6.4-2.sdl7.noarch.rpm
	yum install python-dnf-0.6.4-2.sdl7.noarch.rpm  dnf-0.6.4-2.sdl7.noarch.rpm dnf-conf-0.6.4-2.sdl7.noarch.rpm

As long as typing in these commands,  the dnf could be installed successfully.

	[leo@ASUS-Leo ~]$ cat /etc/redhat-release
	CentOS Linux release 7.4.1708 (Core)
	[leo@ASUS-Leo ~]$ dnf update
	Extra Packages for Enterprise Linux 7 - x86_64  2.5 MB/s |  13 MB     00:05
	CentOS-7 - Base                                 1.9 MB/s | 9.5 MB     00:04
	CentOS-7 - Updates                              2.4 MB/s | 3.2 MB     00:01
	CentOS-7 - Extras                               348 kB/s | 421 kB     00:01
	Using metadata from Thu Sep 21 23:01:58 2017
	Dependencies resolved.
	Nothing to do.
	[leo@ASUS-Leo ~]$
