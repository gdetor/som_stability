#!/usr/bin/bash
# 
# This bash script runs the SOM training algorithm for 10 different PRNG seeds.
# Copyright (C) <2020>  Georgios Is. Detorakis (gdetor@protonmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# [1] "Stability analysis of a neural field self-organizing map",
#      G. Is. Detorakis, A. Chaillet, N.P. Rougier, 2020.

# for i in 10 74 433 721 956 1330 3567 5677 9127 7659
# for i in 654 98 13 4357 2180 236 1008 906 5439 23561
for i in 54 849 113 5089 1234 23986 9865 511 7432 10976
do
    echo "Running seed $i"
    python3 som.py $i 0.3 0.25 $i &
done

wait
echo "All done"
