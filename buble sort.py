import random

class Search(object):
    def __init__(self, List):
        self.List =List
    def buble_sort(self):
        result = []
        for item in self.List:
            if result == []:
                result.append(item)
            else:
                if result[-1]<=item:
                    result.append(item)
                else:
                    for min in result:
                        if min > item:
                            index = result.index(min)
                            result.insert(index, item)
                            break
        return result
search = Search([100, -1, 5, 0, 10,2, 3, -6, 8, 9, 0, 4])
search.buble_sort()