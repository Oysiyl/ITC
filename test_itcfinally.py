#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:44:53 2018.

@author: dmitriy
"""
# import all the necessary libraries
import itcfinally
import unittest


class TestAdd(unittest.TestCase):
    """
    Test the add function from the mymath library
    """

    def test_get_search_query_1(self):
        """
        Test that the addition of two integers returns the correct total
        """
        list2 = ['10-01-2019', '12-01-2019']  # d-m-y
        result = itcfinally.get_search_query(list2, 1)
        # https://itc.ua/?s&after=10-01-2019&before=12-01-2019
        # https://itc.ua/page/3/?s&after=10-01-2019&before=12-01-2019
        list3 = "https://itc.ua/?s&after=10-01-2019&before=12-01-2019"
        self.assertEqual(result, list3)

    def test_get_search_query_2(self):
        """
        Test that the addition of two integers returns the correct total
        """
        list2 = ['10-01-2019', '12-01-2019']  # d-m-y
        result = itcfinally.get_search_query(list2, 2)
        # https://itc.ua/?s&after=10-01-2019&before=12-01-2019
        # https://itc.ua/page/3/?s&after=10-01-2019&before=12-01-2019
        list3 = "https://itc.ua/page/2/?s&after=10-01-2019&before=12-01-2019"
        self.assertEqual(result, list3)

    def test_prep_numbers_1(self):
        """
        Test that the addition of two integers returns the correct total
        """
        count = 1
        result = itcfinally.prep_numbers(count)
        # https://itc.ua/?s&after=10-01-2019&before=12-01-2019
        # https://itc.ua/page/3/?s&after=10-01-2019&before=12-01-2019
        list3 = ["https://itc.ua/"]
        self.assertListEqual(result, list3)

    @unittest.skip('Skip this test')
    def test_prep_numbers_2(self):
        """
        Test that the addition of two integers returns the correct total
        """
        list2 = ['10-01-2019', '12-01-2019']  # d-m-y
        count = 2
        result = itcfinally.prep_numbers_2(count, list2)
        # https://itc.ua/?s&after=10-01-2019&before=12-01-2019
        # https://itc.ua/page/3/?s&after=10-01-2019&before=12-01-2019
        list3 = ["https://itc.ua/?s&after=10-01-2019&before=12-01-2019",
                 "https://itc.ua/page/2/?s&after=10-01-2019&before=12-01-2019"]
        self.assertListEqual(result, list3)


if __name__ == '__main__':
    unittest.main()
