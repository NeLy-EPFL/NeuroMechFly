"""
-----------------------------------------------------------------------
Copyright 2018-2020 Jonathan Arreguit, Shravan Tata Ramalingasetty
Copyright 2018 BioRobotics Laboratory, École polytechnique fédérale de Lausanne

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-----------------------------------------------------------------------

Original code from Jonathan Arreguit and modified by Shravan Tata Ramalingasetty

Code to profile functions in neuromechfly

"""

import pstats
import cProfile


def profile(function, profile_filename='', **kwargs):
    """Profile with cProfile"""
    n_time = kwargs.pop('pstat_n_time', 30)
    n_cumtime = kwargs.pop('pstat_n_cumtime', 30)
    prof = cProfile.Profile()
    result = prof.runcall(function, **kwargs)
    if profile_filename:
        prof.dump_stats(profile_filename)
        pstat = pstats.Stats(profile_filename)
    else:
        pstat = pstats.Stats(prof)
    pstat.sort_stats('time').print_stats(n_time)
    pstat.sort_stats('cumtime').print_stats(n_cumtime)
    return result
