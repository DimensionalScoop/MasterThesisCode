from unittest import TestCase

import scipy.stats
import numpy as np
import pandas as pd


class TestGenerator(TestCase):
    def test_save(self):
        from pandel.generator import Generator

        ds = (50,100,200,)
        ηs = (2,3,4,)
        norm = scipy.stats.norm(loc=100, scale=20)
        ev = ["valid", 100, 4.3, 1000, np.ones((128,))]
        g = Generator(ds, ηs, norm)
        g.events = g.events.append(pd.Series(ev,index = g.columns),
                                   ignore_index = True)
        g.save("test")
        del g

        from_disk = Generator.load("test")

        assert from_disk.ds == ds
        assert from_disk.ηs == ηs
        assert from_disk.norm == norm


    def test_get_description(self):
        self.fail()
