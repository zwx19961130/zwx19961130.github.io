---
layout: post
title: "Chocolatery"
date: 2018-02-18
---


# Chocolatery

发现一个比较好用的windows上的包管理工具：Chocolatery. 
可以在powershell输入

>iex ((new-object net.webclient).DownloadString('https://chocolatey.org/install.ps1'))

或者在cmd输入

>@powershell -NoProfile -ExecutionPolicy unrestricted -Command "iex ((new-object net.webclient).DownloadString('https://chocolatey.org/install.ps1'))" && SET PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin

然后直接回车

[Chocolatey](https://chocolatey.org/)，简单说这就是 Windows 的 apt-get。习惯 Linux 操作方式并非常想用它操纵 Windows 的敬请折腾。Chocolatey 这套包管理系统目前已经包含了近 500 多款常用软件，比如 VLC 神马的。

用法也简单：

查询程序是否在数据库中：clist < 程序名>
安装程序：cinst < 程序名>

跟 Linux 下面包管理比起来天差地别，如果您真的对包管理推崇备至，甚至到了“没有包管理的操作系统是反人类的”，那去找个有包管理的 Linux 发行版最省心，然后记得跟使用“反人类”操作系统的人类做艰苦卓绝的斗争吧！


## 使用

 

举个栗子，你如果想安装7Zip，你可以在命令行输入：

cinst 7Zip

就会自动安装这个压缩软件。

安装Node.js,输入：

cinst node.js

另外还可以安装IE10（Windows 7）：

cinst IE10

安装Visual Studio 2013 Ultimate这个巨无霸也是可以的：

cinst VisualStudio2013Ultimate

软件列表，可以在Chocolatey的软件索引查到。
