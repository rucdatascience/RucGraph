#pragma once

#include <limits>

/*
for a sorted vector<pair<int,T>>, this dunction conducted binary (divide and conquer) operations on this vector;

the int values are unique and sorted from small to large

https://blog.csdn.net/EbowTang/article/details/50770315
*/


template <typename T>
bool sorted_vector_binary_operations_search(std::vector<std::pair<int, T>>& input_vector, int key);

template <typename T>
T sorted_vector_binary_operations_search_weight(std::vector<std::pair<int, T>>& input_vector, int key);


template <typename T>
int sorted_vector_binary_operations_search_position(std::vector<std::pair<int, T>>& input_vector, int key);

template <typename T>
void sorted_vector_binary_operations_erase(std::vector<std::pair<int, T>>& input_vector, int key);

template <typename T>
int sorted_vector_binary_operations_insert(std::vector<std::pair<int, T>>& input_vector, int key, T load);


template <typename T>
bool sorted_vector_binary_operations_search(std::vector<std::pair<int, T>>& input_vector, int key) {

	/*return true if key is in vector; time complexity O(log n)*/

	int left = 0, right = input_vector.size() - 1;

	while (left <= right) {
		int mid = left + ((right - left) / 2); // mid is between left and right (may be equal); 
		if (input_vector[mid].first == key) {
			return true;
		}
		else if (input_vector[mid].first > key) {
			right = mid - 1;
		}
		else {
			left = mid + 1;
		}
	}

	return false;

}

template <typename T>
T sorted_vector_binary_operations_search_weight(std::vector<std::pair<int, T>>& input_vector, int key) {

	/*return std::numeric_limits<T>::max() if key is not in vector; time complexity O(log n)*/

	int left = 0, right = input_vector.size() - 1;

	while (left <= right) {
		int mid = left + ((right - left) / 2); // mid is between left and right (may be equal); 
		if (input_vector[mid].first == key) {
			return input_vector[mid].second;
		}
		else if (input_vector[mid].first > key) {
			right = mid - 1;
		}
		else {
			left = mid + 1;
		}
	}

	return std::numeric_limits<T>::max();

}

template <typename T>
int sorted_vector_binary_operations_search_position(std::vector<std::pair<int, T>>& input_vector, int key) {

	/*return -1 if key is not in vector; time complexity O(log n)*/

	int left = 0, right = input_vector.size() - 1;

	while (left <= right) {
		int mid = left + ((right - left) / 2);
		if (input_vector[mid].first == key) {
			return mid;
		}
		else if (input_vector[mid].first > key) {
			right = mid - 1;
		}
		else {
			left = mid + 1;
		}
	}

	return -1;

}

template <typename T>
void sorted_vector_binary_operations_erase(std::vector<std::pair<int, T>>& input_vector, int key) {

	/*erase key from vector; time complexity O(log n + size()-position ), which is O(n) in the worst case, as
	the time complexity of erasing an element from a vector is the number of elements behind this element*/

	if (input_vector.size() > 0) {
		int left = 0, right = input_vector.size() - 1;

		while (left <= right) {
			int mid = left + ((right - left) / 2);
			if (input_vector[mid].first == key) {
				input_vector.erase(input_vector.begin() + mid);
				break;
			}
			else if (input_vector[mid].first > key) {
				right = mid - 1;
			}
			else {
				left = mid + 1;
			}
		}
	}

}

template <typename T>
int sorted_vector_binary_operations_insert(std::vector<std::pair<int, T>>& input_vector, int key, T load) {

	/*return the inserted position;

	insert <key, load> into vector, if key is already inside, then load is updated; time complexity O(log n + size()-position ), which is O(n) in the worst case, as
	the time complexity of inserting an element into a vector is the number of elements behind this element*/

	int left = 0, right = input_vector.size() - 1;

	while (left <= right) // it will be skept when input_vector.size() == 0
	{
		int mid = left + ((right - left) / 2); // mid is between left and right (may be equal); 
		if (input_vector[mid].first == key) {
			input_vector[mid].second = load;
			return mid;
		}
		else if (input_vector[mid].first > key) {
			right = mid - 1; // the elements after right are always either empty, or have larger keys than input key
		}
		else {
			left = mid + 1; // the elements before left are always either empty, or have smaller keys than input key
		}
	}

	/*the following code is used when key is not in vector, i.e., left > right, specifically, left = right + 1;
	the elements before left are always either empty, or have smaller keys than input key;
	the elements after right are always either empty, or have larger keys than input key;
	so, the input key should be insert between right and left at this moment*/
	input_vector.insert(input_vector.begin() + left, { key,load });
	return left;
}