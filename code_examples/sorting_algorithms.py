"""Sorting Algorithms — merge sort, quicksort, heap sort."""

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    return result + left[i:] + right[j:]

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    lo = [x for x in arr if x < pivot]
    eq = [x for x in arr if x == pivot]
    hi = [x for x in arr if x > pivot]
    return quicksort(lo) + eq + quicksort(hi)

def heap_sort(arr):
    import heapq
    heapq.heapify(arr := list(arr))
    return [heapq.heappop(arr) for _ in range(len(arr))]

# --- demo ---
data = [38, 27, 43, 3, 9, 82, 10]
print("Original: ", data)
print("Merge sort:", merge_sort(data))
print("Quicksort: ", quicksort(data))
print("Heap sort: ", heap_sort(data))
