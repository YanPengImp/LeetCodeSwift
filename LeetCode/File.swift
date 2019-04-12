//
//  File.swift
//  LeetCode
//
//  Created by Imp on 2018/6/28.
//  Copyright © 2018年 jingbo. All rights reserved.
//

import Foundation

class LeetCode {

    public class ListNode {
        public var val: Int
        public var next: ListNode?
        public init(_ val: Int) {
            self.val = val
            self.next = nil
        }
    }

    public class TreeNode {
        public var val: Int
        public var left: TreeNode?
        public var right: TreeNode?
        public init(_ val: Int) {
            self.val = val
            self.left = nil
            self.right = nil
        }
    }

    //1.两数之和
    /*
     * 遍历两个数相加是否为target，或者用target减去一个数判断差值是否在数组中
     */
    func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
        for i in 0..<nums.count {
            for j in i+1..<nums.count{
                if nums[i] + nums[j] == target {
                    return [i,j]
                }
            }
        }
        return [0,0]
    }
    
    //6.Z字形变换
    /*
     * 按照numRows的个数来判断规律，生成二维数组来按列的顺序排列字母，空白处用""填充，最后遍历二维数组得到完整的一个字符串即为结果
     */
    func convert(_ s: String, _ numRows: Int) -> String {
        if s.count <= numRows || numRows == 1 {
            return s
        }
        var matrix = [[String]](repeating: [String](), count: numRows)
        var res = ""
        var arr = [String]()
        for c in s {
            arr.append(String(c))
        }
        let count = (s.count - 1) / (2 * (numRows - 1)) + 1
        for i in 0..<count {
            for j in 0..<numRows {
                let n1 = j + i * 2 * (numRows - 1)
                let n2 = n1 + (numRows - 1 - j) * 2
                if n1 >= arr.count {
                    matrix[j].append("")
                } else {
                    matrix[j].append(arr[n1])
                }
                if j != 0 && j != numRows - 1 {
                    if n2 >= arr.count {
                        matrix[j].append("")
                    } else {
                        matrix[j].append(arr[n2])
                    }
                }
            }
        }
        for i in matrix {
            for j in i {
                res += j
            }
        }
        return res
    }
    //7.反转整数
    /*
     * 从最末位依次得到数字 相加再*10得到最终结果 再判断是否为负数
     */
    func reverse(_ x: Int) -> Int {
        var num = Int(abs(Int(x)))
        var res = 0
        while num > 0 {
            res += num % 10
            num /= 10
            if num > 0 {
                res *= 10
            }
        }
        if res > Int32.max {
            return 0
        }
        return x > 0 ? res : -res
    }
    //9.回文数
    /*
     * 从最末位依次得到数字 相加再*10得到最终结果 再判断是否和原数相等
     */
    func isPalindrome(_ x: Int) -> Bool {
        if x < 0 {
            return false
        }
        var res = 0
        var tmp = x
        while (tmp > 0) {
            res = res * 10 + tmp % 10
            tmp /= 10
        }
        return res == x
    }
    //11. 盛最多水的容器
    /*
     * 长方形面积由长宽决定，取首尾两点为长，高度最小的为高，首比尾高的话就首右移一位，相反的话尾左移一位
     */
    func maxArea(_ height: [Int]) -> Int {
        var res = 0
        var left = 0
        var right = height.count - 1
        while left < right {
            let h = min(height[left], height[right])
            let w = right - left
            res = max(res, h * w)
            if height[left] < height[right] {
                left += 1
            } else {
                right += 1
            }
        }
        return res
    }
    //14.最长公共前缀
    /*
     * 先根据字符串的长度排序，最前面的肯定是包含最长公共前缀，依次添加一个字符然后遍历后面的是否都包含前面的前缀
     */
    func longestCommonPrefix(_ strs: [String]) -> String {
        let arr = strs.sorted { (s1, s2) -> Bool in
            return s1.count < s2.count
        }
        var string = ""
        if arr.isEmpty {
            return string
        }
        for i in arr.first! {
            string.append(i)
            for s in arr {
                if s.hasPrefix(string) == false {
                    return String(string.dropLast())
                }
            }
        }
        return string
    }
    //21.合并两个有序链表
    /*
     * 分别得到两个链表的值的数组，再组合之后排序，重新生成一个顺序链表
     */
    func mergeTwoLists(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        func numsArrOfList(_ list: ListNode?) -> [Int] {
            var arr = [Int]()
            var l = list
            while (l != nil) {
                arr.append(l!.val)
                l = l?.next
            }
            return arr
        }
        let arr1 = numsArrOfList(l1)
        let arr2 = numsArrOfList(l2)
        var arr = arr1 + arr2
        arr.sort()
        guard !arr.isEmpty else {
            return nil
        }
        var list = ListNode.init(arr.last!)
        for i in 1..<arr.count {
            let l = ListNode.init(arr[arr.count-i-1])
            l.next = list
            list = l
        }
        return list
    }
    //26.移除数组中重复元素
    /*
     * 因为是顺序数组，所以正常的没有重复元素的数组应该是从小到大依次只出现依次，判断前后元素是否相等，相等则移除
     */
    func removeDuplicates(_ nums: inout [Int]) -> Int {
        var i = 1
        var n = nums.count
        while i < n {
            if nums[i] == nums[i-1] {
                nums.remove(at: i)
                n -= 1
            } else {
                i += 1
            }
        }
        return nums.count
    }
    //27.移除元素
    /*
     * 直接遍历是否等于当前val，相等则移除
     */
    func removeElement(_ nums: inout [Int], _ val: Int) -> Int {
        var i = 0
        var n = nums.count
        while(i < n) {
            if nums[i] == val {
                nums.remove(at:i)
                n -= 1
            } else {
                i += 1
            }
        }
        return nums.count
    }
    //34.搜索范围 二分法搜索
    /*
     * 二分搜索法逐步减小范围
     */
    func searchRange(_ nums: [Int], _ target: Int) -> [Int] {
        var left = 0
        var right = nums.count - 1
        func searchRange(_ left: inout Int, _ right: inout Int,_ nums: [Int], _ target: Int) {
            if left > right {
                return
            }
            let mid = (left + right) / 2
            if nums[mid] < target {
                left = mid + 1
                searchRange(&left, &right, nums, target)
            } else if nums[mid] > target {
                right = mid - 1
                searchRange(&left, &right, nums, target)
            } else {
                while left >= 0 && nums[left] == target {
                    left -= 1
                }
                if left < 0 || nums[left] != target {
                    left += 1
                }
                while right <= nums.count - 1 && nums[right] == target {
                    right += 1
                }
                if right > nums.count - 1 || nums[right] != target {
                    right -= 1
                }
            }
        }
        while left <= right {
            searchRange(&left, &right, nums, target)
            if left > right {
                return [-1,-1]
            }
            if nums[left] == nums[right] && nums[left] == target {
                break
            }
        }
        if left <= right {
            return [left,right]
        } else {
            return [-1,-1]
        }
    }
    //36.有效的数独
    //分三步 1、是否在当前行有该数字
    //      2、是否在当前列有该数字
    //      3、是否在当前小的九宫格有该数字
    //还需要判断行列和不当前位置相等
    func isValidSudoku(_ board: [[Character]]) -> Bool {
        func isValid(_ board: [[Character]],_ row: Int,_ col: Int,_ c: Character) -> Bool {
            func isExistInRow(_ board: [[Character]],_ row: Int,_ col: Int, _ c: Character) -> Bool {
                for i in 0..<board.count {
                    if board[row][i] == c && i != col{
                        return true
                    }
                }
                return false
            }
            func isExistInCol(_ board: [[Character]],_ row: Int,_ col: Int, _ c: Character) -> Bool {
                for i in 0..<board.count {
                    if board[i][col] == c && i != row{
                        return true
                    }
                }
                return false
            }
            func isExistInSmallGird(_ board: [[Character]],_ startRow: Int, _ startCol: Int,_ row: Int,_ col: Int,_ c: Character) -> Bool {
                for i in 0..<3 {
                    for j in 0..<3 {
                        if board[i + startRow][j + startCol] == c && i + startRow != row && j + startCol != col {
                            return true
                        }
                    }
                }
                return false
            }
            let startRow = row - row % 3
            let startCol = col - col % 3
            return !isExistInCol(board, row, col, c) && !isExistInRow(board, row, col, c) && !isExistInSmallGird(board, startRow, startCol, row, col, c)
        }
        for i in 0..<9 {
            for j in 0..<9 {
                if board[i][j] != Character.init(".") {
                    if isValid(board, i, j, board[i][j]) == false {
                        return false
                    }
                }
            }
        }
        return true
    }
    //37.解数独
    func solveSudoku(_ board: inout [[Character]]) {
        //判断数字是否可以填入该格子
        //分三步 1、是否在当前行有该数字
        //      2、是否在当前列有该数字
        //      3、是否在当前小的九宫格有该数字
        func isValid(_ board: [[Character]],_ row: Int,_ col: Int,_ num: Int) -> Bool {
            func isExistInRow(_ board: [[Character]],_ row: Int, _ num: Int) -> Bool {
                for i in 0..<board.count {
                    if board[row][i] == Character.init("\(num)") {
                        return true
                    }
                }
                return false
            }
            func isExistInCol(_ board: [[Character]],_ col: Int, _ num: Int) -> Bool {
                for i in 0..<board.count {
                    if board[i][col] == Character.init("\(num)") {
                        return true
                    }
                }
                return false
            }
            func isExistInSmallGird(_ board: [[Character]],_ startRow: Int, _ startCol: Int, num: Int) -> Bool {
                for i in 0..<3 {
                    for j in 0..<3 {
                        if board[i + startRow][j + startCol] == Character.init("\(num)") {
                            return true
                        }
                    }
                }
                return false
            }
            let startRow = row - row % 3
            let startCol = col - col % 3
            return !isExistInCol(board, col, num) && !isExistInRow(board, row, num) && !isExistInSmallGird(board, startRow, startCol, num: num)
        }
        //判断未填入数字的位置 最后一个Bool表示所有位置都填写完毕，当为true时结束
        func getEmptyPosition(_ board: [[Character]]) -> (Int,Int,Bool) {
            for i in 0..<board.count {
                for j in 0..<board.first!.count {
                    if board[i][j] == Character.init(".") {
                        return (i,j,false)
                    }
                }
            }
            return (0,0,true)
        }
        func fillSudoku(_ board: inout [[Character]]) -> Bool {
            let position = getEmptyPosition(board)
            if position.2 == true {
                return true
            }
            let row = position.0
            let col = position.1
            for i in 1...9 {
                if isValid(board, row, col, i) {
                    //填入该数字
                    board[row][col] = Character.init("\(i)")
                    //如果继续往下能够执行完成 直接返回true了
                    if fillSudoku(&board) {
                        return true
                    }
                    //回溯算法：如果没有能够执行完的话 会把之前的值回退为"." 然后下一次找未填入的数字又重新从该位置开始了
                    board[row][col] = Character.init(".")
                }
            }
            return false
        }
        let _ = fillSudoku(&board)
        print(board)
    }
    //38.报数
    //下一个数都是由前一个数决定
    //遍历看下一个字符是否与前一个相等，注意count的变化 还有末尾临界值的判断
    func countAndSay(_ n: Int) -> String {
        func countToString(_ s: String) -> String {
            var pos: Character = s.first!
            var count = 0
            var str = ""
            for (index,c) in s.enumerated() {
                if c != pos {
                    str.append("\(count)")
                    str.append(pos)
                    pos = c
                    count = 1
                } else {
                    count += 1
                }
                if index == s.count - 1 {
                    str.append("\(count)")
                    str.append(c)
                }
            }
            return str;
        }
        if n == 1 {
            return "1"
        } else {
            return countToString(countAndSay(n-1))
        }
    }
    //41.缺失的第一个正整数
    func firstMissingPositive(_ nums: [Int]) -> Int {
        let newArr = nums.filter { (n) -> Bool in
            return n > 0
        }.sorted()
        var min = 0
        for num in newArr {
            if num - min > 1 {
                return min + 1
            }
            min = num
        }
        return newArr.last ?? 0 + 1
    }
    //45.跳跃游戏II
    func jump(_ nums: [Int]) -> Int {
        var res = 0
        var cur = 0
        var i = 0
        while cur < nums.count - 1 {
            res += 1
            let pre = cur
            while i <= pre {
                cur = max(cur, i + nums[i])
                i += 1
            }
        }
        return res
    }
    //54.螺旋矩阵  TODO
    func spiralOrder(_ matrix: [[Int]]) -> [Int] {
        var res = [Int]()
        var m = matrix.count//行
        var n = matrix[0].count//列
        func rotateArray(_ start: Int,_ end: Int) {
            if start >= end {
                return
            }
        }
        return res
    }
    //55.跳跃游戏
    func canJump(_ nums: [Int]) -> Bool {
        func canJumpToIndex(_ nums: [Int], _ index: Int) -> Bool {
            var i = index - 1
            while i >= 0 && nums[i] < index - i  {
                i -= 1
            }
            if i < 0 {
                return false
            } else if i == 0 {
                return nums[0] >= index
            } else {
                return canJumpToIndex(nums, i)
            }
        }
        if nums.count > 1 {
            return canJumpToIndex(nums, nums.count - 1)
        } else {
            return true
        }
    }
    //58. 最后一个单词的长度
    func lengthOfLastWord(_ s: String) -> Int {
        var newS = s
        while newS.count > 0 && newS.last! == " " {
            newS = String.init(newS.dropLast())
        }
        if newS.count == 0 {
            return 0
        }
        let arr = newS.components(separatedBy: " ")
        return arr.last!.count
    }
    //60.第K个排列
    func getPermutation(_ n: Int, _ k: Int) -> String {
        var arr = [Int]()
        for i in 0..<n {
            arr.append(i + 1)
        }
        var tmp: [Int] = [Int]()
        func getFactorial(n: Int) -> Int {
            if n < 2 {
                return 1
            } else {
                return n * getFactorial(n: n - 1)
            }
        }
        var tmpN = n
        var tmpK = k
        while tmpN > 0 {
            let count = getFactorial(n: tmpN - 1)
            tmpN -= 1
            var index = tmpK / count
            tmpK = tmpK % count
            if index > 0 && tmpK == 0 {
                index -= 1
            }
            tmp.append(arr[index])
            arr.remove(at: index)
            if tmpK == 0  {
                tmp += arr.reversed()
                tmpN = 0
                break
            }
        }
        var str = ""
        for i in tmp {
            str += String(format:"%zd",i)
        }
        return str
    }
    //62.不同路径
    /*
     * 到达某个位置必经该位置的左边或者上边一格位置，递归成计算达到两个上一步位置的和，用数组保存了某些位置的值，免于多次计算
     * 也可以用dp方法，下一步由上一步决定
     */
    func uniquePaths(_ m: Int, _ n: Int) -> Int {
        var arr = [[Int]](repeating: [Int](repeating: 0, count: n), count: m)
        func newUniquePaths(_ m: Int, _ n: Int) -> Int {
            if m == 1 || n == 1 {
                return 1
            }
            var a = arr[m-1-1][n-1]
            var b = arr[m-1][n-1-1]
            if a == 0 {
                arr[m-1-1][n-1] = newUniquePaths(m-1,n)
                a = newUniquePaths(m-1,n)
            }
            if b == 0 {
                arr[m-1][n-1-1] = newUniquePaths(m,n-1)
                b = newUniquePaths(m,n-1)
            }
            return a + b
        }
        return newUniquePaths(m,n)
    }
    //63.不同路径 II
    /*
     * 先确定第一排和第一列的情况，然后用dp方法，每一个位置都取决于上面和左边的情况，需要排除他们为1，自己为1的情况
     */
    func uniquePathsWithObstacles(_ obstacleGrid: [[Int]]) -> Int {
        var dp = [[Int]](repeating: [Int](repeating: 0, count: obstacleGrid[0].count), count: obstacleGrid.count)
        //确定第一列
        for i in 0..<obstacleGrid.count {
            if obstacleGrid[i][0] == 0 {
                dp[i][0] = 1
            }
            if obstacleGrid[i][0] == 1 {
                for j in 0..<dp.count {
                    if j < i {
                        dp[j][0] = 1
                    }
                }
                break
            }
        }
        //确定第一排
        for i in 0..<obstacleGrid[0].count {
            if obstacleGrid[0][i] == 0 {
                dp[0][i] = 1
            }
            if obstacleGrid[0][i] == 1 {
                for j in 0..<dp[0].count {
                    if j < i {
                        dp[0][j] = 1
                    }
                }
                break
            }
        }
        for i in 1..<obstacleGrid.count {
            for j in 1..<obstacleGrid[0].count {
                if obstacleGrid[i-1][j] == 0 && obstacleGrid[i][j] == 0 {
                    dp[i][j] += dp[i-1][j]
                }
                if obstacleGrid[i][j-1] == 0 && obstacleGrid[i][j] == 0 {
                    dp[i][j] += dp[i][j-1]
                }
            }
        }
        return dp[obstacleGrid.count-1][obstacleGrid[0].count-1]
    }
    //64.最小路径和 dp方法
    /*
     * 动态规划 走到每一个位置所需的步数都由上一步所决定
     */
    func minPathSum(_ grid: [[Int]]) -> Int {
        let n = grid.count
        let m = grid[0].count
        var dp = [[Int]](repeating: [Int](repeating: 0, count: m), count: n)
        dp[0][0] = grid[0][0]
        for i in 1..<m {
            dp[0][i] = dp[0][i-1] + grid[0][i]
        }
        for i in 1..<n {
            dp[i][0] = dp[i-1][0] + grid[i][0]
        }
        for i in 1..<n {
            for j in 1..<m {
                if dp[i-1][j] < dp[i][j-1] {
                    dp[i][j] = dp[i-1][j] + grid[i][j]
                } else {
                    dp[i][j] = dp[i][j-1] + grid[i][j]
                }
            }
        }
        return dp[n-1][m-1]
    }
    //66.加一
    /*
     * 最后一位加1如果之前是9，最后一位变成0，前一位再加1，依次类推。
     */
    func plusOne(_ digits: [Int]) -> [Int] {
        guard digits.count > 0 else {
            return digits
        }
        var arr = digits
        var i = arr.count - 1
        arr[i] = arr[i] + 1
        while i >= 0 {
            if arr[i] > 9 {
                arr[i] = 0
                if i == 0 {
                    arr.insert(1, at: 0)
                    return arr
                }
                arr[i-1] = arr[i-1] + 1
                i -= 1
            } else {
                break
            }
        }
        return arr
    }
    //69.x的平方根
    /*
     * 取中间值去算平方和x作比较，等到最终res * res == x时候返回
     */
    func mySqrt(_ x: Int) -> Int {
        var left = 0
        var right = x
        var res = 0
        while left < right {
            if right * right == x {
                return right
            }
            res = (left + right) / 2
            if res * res == x || (res * res < x && (res + 1)*(res + 1) > x) {
                return res
            } else if res * res < x {
                left = res
            } else {
                right = res
            }
        }
        return res
    }
    //70.爬楼梯
    /*
     * 爬n层可以是在n-1层的时候爬一步或者是在n-2层的时候爬2步到达，所以n层是n-1和n-2层的和
     */
    func climbStairs(_ n: Int) -> Int {
        if n < 3 {
            return n
        }
        var arr = [Int]()
        arr.append(1)
        arr.append(2)
        for i in 2..<n {
            arr.append(arr[i-1]+arr[i-2])
        }
        return arr[n-1]
    }
    //75.颜色分类
    func sortColors(_ nums: inout [Int]) {
        var left = 0
        var index = 0
        var right = nums.count - 1
        while index <= right {
            if nums[index] == 0 {
                nums.swapAt(index, left)
                left += 1
            } else if nums[index] == 2 {
                nums.swapAt(index, right)
                index -= 1
                right -= 1
            }
            index += 1
        }
        print(nums)
    }
    //78.子集
    //这种方法有点麻烦 是做了一个全组合的汇总
    func subsets(_ nums: [Int]) -> [[Int]] {
        func subsetsWithCount(_ nums: [Int], _ count: Int) -> [[Int]] {
            var res = [[Int]]()
            var sortArray = [Int](repeating: 0, count: count)
            func combineMatch(array: [Int], nLength: Int, m: Int, storeArray: inout [Int], outLength: Int, outArray: inout [[Int]]) {
                if m == 0{
                    outArray.append(storeArray)
                } else {
                    for index in stride(from: nLength, to: m - 1, by: -1) {
                        storeArray[m-1] = array[index - 1]
                        combineMatch(array: array, nLength: index - 1, m: m - 1, storeArray: &storeArray, outLength: outLength, outArray: &outArray)
                    }
                }
            }
            combineMatch(array: nums, nLength: nums.count, m: count, storeArray: &sortArray, outLength: count, outArray: &res)
            return res
        }
        var res = [[Int]]()
        for i in 0...nums.count {
            res += subsetsWithCount(nums, i)
        }
        return res
    }
    //88.合并两个有序数组
    func merge(_ nums1: inout [Int], _ m: Int, _ nums2: [Int], _ n: Int) {
        var count = m + n - 1
        var i = m - 1
        var j = n - 1
        while (i >= 0 && j >= 0) {
            if nums1[i] > nums2[j] {
                nums1[count] = nums1[i]
                i -= 1
            } else {
                nums1[count] = nums2[j]
                j -= 1
            }
            count -= 1
        }
        while (j >= 0) {
            nums1[count] = nums2[j]
            j -= 1
            count -= 1
        }
    }
    //98.验证二叉搜索树
    func isValidBST(_ root: TreeNode?) -> Bool {
        func isTrue(_ node: TreeNode?, _ max: Int, _ min: Int) -> Bool {
            if node == nil {
                return true
            }
            if (node?.val)! < min || (node?.val)! > max {
                return false
            }
            return isTrue(node?.left, (node?.val)!, min) && isTrue(node?.right, max, (node?.val)!)
        }
        return isTrue(root, Int.max, Int.min)
    }
    //100.相同的树
    func isSameTree(_ p: TreeNode?, _ q: TreeNode?) -> Bool {
        if p?.val != q?.val {
            return false
        }
        if p?.left == nil && q?.left == nil && p?.right == nil && q?.right == nil {
            return true
        }
        return isSameTree(p?.left,q?.left) && isSameTree(p?.right,q?.right)
    }
    //101.对称二叉树
    func isSymmetric(_ root: TreeNode?) -> Bool {
        func isTrue(_ left: TreeNode?, _ right: TreeNode?) -> Bool {
            if left?.val != right?.val {
                return false
            } else {
                if (left?.left == nil && right?.right == nil) && (left?.right == nil && right?.left == nil) {
                    return true
                }
                return isTrue(left?.left, right?.right) && isTrue(left?.right, right?.left)
            }
        }
        return isTrue(root?.left, root?.right)
    }
    //104.二叉树的最大深度
    func maxDepth(_ root: TreeNode?) -> Int {
        var res = 0
        if root != nil {
            let leftLength = maxDepth(root?.left)
            let rightLenght = maxDepth(root?.right)
            res = leftLength > rightLenght ? leftLength + 1 : rightLenght + 1
        }
        return res
    }
    //111.二叉树的最小深度
    func minDepth(_ root: TreeNode?) -> Int {
        if root == nil {
            return 0
        }
        let leftLength = minDepth(root?.left)
        let rightLenght = minDepth(root?.right)
        if leftLength == 0 || rightLenght == 0 {
            return leftLength + rightLenght + 1
        }
        return min(leftLength, rightLenght) + 1
    }
    //112.路径总和
    func hasPathSum(_ root: TreeNode?, _ sum: Int) -> Bool {
        if root == nil {
            return false
        }
        if root?.val == sum && root?.left == nil && root?.right == nil{
            return true
        }
        return hasPathSum(root?.left,sum - (root?.val)!) || hasPathSum(root?.right,sum - (root?.val)!)
    }
    //118.杨辉三角
    func generate(_ numRows: Int) -> [[Int]] {
        var res = [[Int]]()
        if numRows >= 1 {
            res.append([1])
            for i in 1..<numRows {
                var row = [Int]()
                row.append(1)
                for j in 1...i {
                    if j == i {
                        row.append(1)
                    } else {
                        row.append(res[i-1][j-1] + res[i-1][j])
                    }
                }
                res.append(row)
            }
        }
        return res
    }
    //119.杨辉三角2
    //数学公式
    func getRow(_ rowIndex: Int) -> [Int] {
        var res = [Int](repeating: 1, count: rowIndex + 1)
        res[0] = 1
        res[rowIndex] = 1
        for i in 1..<(res.count + 1)/2 {
            res[i] = res[i-1] * (rowIndex - i + 1) / i
            res[rowIndex-i] = res[i-1] * (rowIndex - i + 1) / i
        }
        return res
    }
    //121.买卖股票最佳时机
    func maxProfit(_ prices: [Int]) -> Int {
        if prices.isEmpty {
            return 0
        }
        var result = 0
        var cur = prices[0]
        for i in prices {
            if i < cur {
                cur = i
            } else {
                let tmp = i - cur
                if tmp > result {
                    result = tmp
                }
            }
        }
        return result
    }
    //134.加油站
    func canCompleteCircuit(_ gas: [Int], _ cost: [Int]) -> Int {
        let count = gas.count
        for i in 0..<count {
            if gas[i] < cost[i] {
                continue
            }
            var tmp = 0
            for j in 0..<count {
                let index = (i + j) % count
                tmp += gas[index] - cost[index]
                if tmp < 0 {
                    break
                }
            }
            if tmp >= 0 {
                return i
            }
        }
        return -1
    }
    //135.分发糖果
    func candy(_ ratings: [Int]) -> Int {
        var result = [Int](repeating: 1, count: ratings.count)
        for i in 1..<ratings.count {
            if ratings[i] > ratings[i-1] && result[i] <= result[i-1] {
                result[i] = result[i-1] + 1
            }
        }
        for i in stride(from: ratings.count - 2, to: 0, by: -1) {
            if ratings[i-1] > ratings[i] && result[i-1] <= result[i] {
                result[i-1] = result[i] + 1
            }
        }
        return result.reduce(0, +)
    }
    //136.只出现一次的数字
    func singleNumber(_ nums: [Int]) -> Int {
        var newNums = nums
        newNums.sort()
        for index in stride(from: 0, to: newNums.count, by: 2) {
            if index == newNums.count - 1 || index == newNums.count - 2 {
                return newNums[index]
            }
            if newNums[index] != newNums[index + 1] {
                return newNums[index]
            }
        }
        return 0
    }
    //152.乘积最大子序列
    func maxProduct(_ nums: [Int]) -> Int {
        var mn = nums.first!
        var mx = nums.first!
        var result = nums.first!
        for i in 1..<nums.count {
            let tmax = mx
            let tmin = mn
            mx = max(max(nums[i], tmax * nums[i]), tmin * nums[i])
            mn = min(min(nums[i], tmax * nums[i]), tmin * nums[i])
            result = max(result, mx)
        }
        return result
    }
    //162.寻找峰值
    func findPeakElement(_ nums: [Int]) -> Int {
        if nums.count == 1 {
            return 0
        }
        var left = 0
        var right = nums.count - 1
        if nums[0] > nums[1] {
            return 0
        }
        if nums[right] > nums[right-1] {
            return right
        }
        while left < right {
            left += 1
            right -= 1
            if nums[left] > nums[left-1] && nums[left] > nums[left+1] {
                return left
            }
            if nums[right] > nums[right-1] && nums[right] > nums[right+1] {
                return right
            }
        }
        return 0
    }
    //165.比较版本号
    func compareVersion(_ version1: String, _ version2: String) -> Int {
        var version1Arr = version1.components(separatedBy: ".").map { (s) -> Int in
            return Int(s)!
        }
        var version2Arr = version2.components(separatedBy: ".").map { (s) -> Int in
            return Int(s)!
        }
        while version1Arr.last == 0 {
            version1Arr.removeLast()
        }
        while version2Arr.last == 0 {
            version2Arr.removeLast()
        }
        let count = min(version1Arr.count, version2Arr.count)
        if count == 0 {
            if version1Arr.count < version2Arr.count {
                return -1
            } else if version1Arr.count > version2Arr.count {
                return 1
            } else {
                return 0
            }
        }
        for i in 0..<count {
            if version1Arr[i] > version2Arr[i] {
                return 1
            } else if version1Arr[i] < version2Arr[i] {
                return -1
            } else {
                if i == count - 1 {
                    if version1Arr.count == version2Arr.count {
                        return 0
                    } else {
                        return count == version1Arr.count ? -1 : 1
                    }
                }
            }
        }
        return 0
    }
    //166.分数到小数
    func fractionToDecimal(_ numerator: Int, _ denominator: Int) -> String {
        if numerator % denominator == 0 {
            return "\(numerator/denominator)"
        }

        func numCount(num: Int) -> Int {
            var count = 0
            var new = num
            while new > 0 {
                new /= 10
                count += 1
            }
            return count
        }

        var res = ""
        var newNumerator = Int(abs(numerator))
        let newDenominator = Int(abs(denominator))
        var index = 0
        var hasDot = false
        //保存之前出现的过除数，寻找循环
        var dict = [Int:Int]()
        while true {
            if let i = dict[newNumerator] {
                res.insert("(", at: String.Index.init(encodedOffset: i))
                res += ")"
                break;
            }
            dict[newNumerator] = index
            let dealer = newNumerator / newDenominator
            if dealer == 0 {
                if res.count == 0 {
                    res += "0."
                    hasDot = true
                    index += 2
                } else {
                    res += "0"
                    index += 1
                }
            } else {
                res += "\(dealer)"
                index += numCount(num: dealer)
                newNumerator = newNumerator % newDenominator
            }
            if newNumerator == 0 {
                break
            }
            if !hasDot && newNumerator < newDenominator {
                res += "."
                index += 1
                hasDot = true
            }
            newNumerator *= 10
        }
        if numerator * denominator < 0 {
            res.insert("-", at: String.Index.init(encodedOffset: 0))
        }
        return res
    }
    //167. 两数之和 II - 输入有序数组
    func twoSum2(_ numbers: [Int], _ target: Int) -> [Int] {
        var left = 0
        var right = numbers.count - 1
        while  left < right{
            if numbers[left] + numbers[right] > target {
                right -= 1
            } else if numbers[left] + numbers[right] < target {
                left += 1
            } else {
                return [left + 1,right + 1]
            }
        }
        return [left + 1]
    }
    //169.求众数
    func majorityElement(_ nums: [Int]) -> Int {
        var dict = [Int:Int]()
        for i in nums {
            if let count = dict[i] {
                dict[i] = count + 1
            } else {
                dict[i] = 1
            }
            if dict[i]! > nums.count / 2 {
                return i
            }
        }
        return 0
    }
    //172.阶乘末尾的0
    func trailingZeroes(_ n: Int) -> Int {
        var count = 0
        var new = n
        while new >= 5 {
            new /= 5
            count += new
        }
        return count
    }
    //179.最大数
    func largestNumber(_ nums: [Int]) -> String {
        var tmp = [String]()
        let newStrs = nums.map { (i) -> String in
            String(format:"%ld",i)
        }
        tmp = newStrs.sorted { (s1, s2) -> Bool in
            return (s1 + s2).compare((s2 + s1)) == ComparisonResult.orderedDescending
        }
        if tmp.first! == "0" {
            return "0"
        }
        var string = ""
        for i in tmp {
            string += i
        }
        return string
    }
    //202.快乐数 每位数平方和相加 一直到是否可以为1
    func isHappy(_ n: Int) -> Bool {
        func getNext(_ num: Int) -> Int {
            var tmp = num
            var res = 0
            while(tmp > 0) {
                let x = tmp % 10
                tmp /= 10
                res += x * x
            }
            return res
        }
        var set:Set<Int> = [0]
        var m = n
        while(true) {
            m = getNext(m)
            if m == 1 {
                return true
            } else if set.contains(m) {
                return false
            } else {
                set.insert(m)
            }
        }
    }
    //203.删除链表中的节点 *****编译出错******
    func removeElements(_ head: ListNode?, _ val: Int) -> ListNode? {
        var p = head
        var next = head
        while next != nil {
            if next?.val == val {
                p?.next = next?.next
            } else {
                p = next
            }
            next = next?.next
        }
        return head
    }
    //204.质数个数
    func countPrimes(_ n: Int) -> Int {
        if n < 3 {
            return 0
        }
        if n == 3 {
            return 1
        }
        var res = 0
        var primes = [Bool](repeating: true, count: n)
        primes[0] = false
        primes[1] = false
        let max = Int(sqrt(Double(n)))
        for i in 2...max {
            if primes[i] {
                //如果这个数是质数 那么他的倍数都是合数
                for j in stride(from: 2 * i, to: n, by: i) {
                    primes[j] = false
                }
            }
        }
        for i in primes {
            if i {
                res += 1
            }
        }
        return res
    }
    //215.数组中第k大元素
    func findKthLargest(_ nums: [Int], _ k: Int) -> Int {
        let arr = nums.sorted(by: >)
        return arr[k - 1]
    }
    //231.2的幂
    func isPowerOfTwo(_ n: Int) -> Bool {
        if n == 1 || n == 2 {
            return true
        }
        if n % 2 != 0 {
            return false
        }
        let a = n % 10
        if a != 2 && a != 4 && a != 8 && a != 6 {
            return false
        }
        return isPowerOfThree(n/2)
    }

    //238.除以自身以外的数组的乘积
    func productExceptSelf(_ nums: [Int]) -> [Int] {
        let count = nums.count
        var leftArr = [Int](repeating: 1, count: count)
        var rightArr = [Int](repeating: 1, count: count)
        for index in 0..<count {
            if index == 0 {
                leftArr[index] = 1
                rightArr[count - 1 - index] = 1
            } else {
                leftArr[index] = leftArr[index - 1] * nums[index - 1]
                rightArr[count - 1 - index] = rightArr[count - index] * nums[count - index]
            }
        }
        var result = [Int]()
        for index in 0..<count {
            result.append(leftArr[index]*rightArr[index])
        }
        return result
    }
    //239.滑动窗口最大值
    func maxSlidingWindow(_ nums: [Int], _ k: Int) -> [Int] {
        if nums.count == 0 {
            return []
        }
        var res = [Int]()
        var arr = [Int]()
        for i in 0..<k {
            arr.append(nums[i])
        }
        var max = arr.max()!
        res.append(max)
        for i in k..<nums.count {
            if nums[i] >= max {
                max = nums[i]
            }
            arr.remove(at: 0)
            arr.append(nums[i])
            if nums[i-k] >= max {
                max = arr.max()!
            }
            res.append(max)
        }
        return res
    }
    //257.二叉树的所有路径
    func binaryTreePaths(_ root: TreeNode?) -> [String] {
        var path = [String]()
        func allPath(_ tree: TreeNode?, _ path: inout [String], _ val: String) {
            if tree?.left == nil && tree?.right == nil {
                path.append(val)
                return
            }
            if tree?.left != nil {
                allPath(tree!.left!, &path, val + "->\(tree!.left!.val)")
            }
            if tree?.right != nil {
                allPath(tree!.right!, &path, val + "->\(tree!.right!.val)")
            }
        }
        if root != nil {
            allPath(root, &path, "\(root!.val)")
        }
        return path
    }
    //258.各位相加
    func addDigits(_ num: Int) -> Int {
        return (num - 1) % 9 + 1
    }
    //263.丑数
    func isUgly(_ num: Int) -> Bool {
        if num == 0 {
            return false
        }
        if num == 1 {
            return true
        }
        if num % 2 == 0 {
            return isUgly(num / 2)
        }
        if num % 3 == 0 {
            return isUgly(num / 3)
        }
        if num % 5 == 0 {
            return isUgly(num / 5)
        }
        return false
    }
    //268.缺失数字
    //可以直接取^i^nums[i] 得最终结果
    func missingNumber(_ nums: [Int]) -> Int {
        var sum = 0
        for (index,num) in nums.enumerated() {
            sum += (index - num)
        }
        return nums.count + sum
    }
    //283.移动零
    func moveZeroes(_ nums: inout [Int]) {
        var cur = 0
        for i in nums {
            if i != 0 {
                nums[cur] = i
                cur += 1
            }
        }
        for i in cur..<nums.count {
            nums[i] = 0
        }
    }
    //292.Nim游戏
    func canWinNim(_ n: Int) -> Bool {
        return n % 4 != 0
    }
    //323.3的幂
    func isPowerOfThree(_ n: Int) -> Bool {
        if n == 1 || n == 3{
            return true
        }
        if n % 3 != 0 {
            return false
        }
        var num = n
        let a = n % 10
        if a != 1 && a != 3 && a != 9 && a != 7 {
            return false
        }
        var sum = 0
        while num > 0 {
            sum += num % 10
            num /= 10
        }
        if sum % 3 != 0 {
            return false
        }
        return isPowerOfThree(n/3)
    }
    //334.递增三元子序列
    func increasingTriplet(_ nums: [Int]) -> Bool {
        var first = Int.max
        var second = Int.max
        for i in nums {
            if i < first {
                first = i
                continue
            }
            if i > first && i < second {
                second = i
                continue
            }
            if i > second {
                return true
            }
        }
        return false
    }
    //342.4的幂
    func isPowerOfFour(_ num: Int) -> Bool {
        if num < 0 || num & (num - 1) != 0 {
            return false
        }
        return num & 0x55555555 != 0
    }
    //343.整数拆分
    func integerBreak(_ n: Int) -> Int {
        if n > 6 {
            return integerBreak(n-3) * 3
        }
        if n == 3 {
            return 2
        }
        if n == 4 {
            return 4
        }
        if n == 5 {
            return 6
        }
        if n == 6 {
            return 9
        }
        return 1
    }
    //344.反转字符串
    func reverseString(_ s: String) -> String {
        var arr = [Character]()
        for i in s {
            arr.append(i)
        }
        arr = arr.reversed()
        var string = ""
        for i in arr {
            string.append(i)
        }
        return string
    }
    //357. 计算各个位数不同的数字个数
    func countNumbersWithUniqueDigits(_ n: Int) -> Int {
        if n == 0 {
            return 1
        }
        var count = 10
        var add = 9
        for i in 1..<n {
            if i == 10 {
                break
            }
            count += add * (10 - i)
            add *= (10 - i)
        }
        return count
    }
    //365.水壶问题
    func canMeasureWater(_ x: Int, _ y: Int, _ z: Int) -> Bool {
        if x + y == z || x == z || y == z {
            return true
        }
        if x + y < z {
            return false
        }
        func gcd(_ a: Int, _ b: Int) -> Int{
            if b == 0 {
                return a
            }
            return gcd(b, a%b)
        }
        func swap(_ a: inout Int, _ b: inout Int) {
            if a < b {
                let tmp = a
                a = b
                b = tmp
            }
        }
        if x != 0 && y != 0 {
            var a = x
            var b = y
            swap(&a,&b)
            if z % gcd(a,b) == 0 {
                return true
            }
            return false
        }
        return false
    }
    //367. 有效的完全平方数
    func isPerfectSquare(_ num: Int) -> Bool {
        var i = 1
        var res = num
        while (res > 0) {
            res -= i
            i += 2
        }
        return res == 0
    }
    //372.超级次方
    func superPow(_ a: Int, _ b: [Int]) -> Int {
        func power(_ x: Int, _ n: Int) -> Int {
            if (n == 0) {
                return 1
            }
            if (n == 1) {
                return x % 1337
            }
            return power(x % 1337, n / 2) * power(x % 1337, n - n / 2) % 1337
        }
        var res = 1
        for i in b {
            res = power(res, 10) * power(a, i) % 1337;
        }
        return res % 1337
    }
    //387. 字符串中的第一个唯一字符
    func firstUniqChar(_ s: String) -> Int {
        var dict = [Character:Int]()
        for str in s {
            if let c = dict[str] {
                dict[str] = c + 1
            } else {
                dict[str] = 1
            }
        }
        var arr = [Character]()
        for (key,value) in dict {
            if value != 1 {
                continue
            }
            arr.append(key)
        }
        for (index,str) in s.enumerated() {
            for c in arr {
                if str == c {
                    return index
                }
            }
        }
        return -1
    }
    //390.消除游戏
    func lastRemaining(_ n: Int) -> Int {
        return n == 1 ? 1 : 2 * (n/2 + 1 - lastRemaining(n/2))
    }
    //397.整数替换
    func integerReplacement(_ n: Int) -> Int {
        if n == 1 {
            return 0
        }
        if n == 2 {
            return 1
        }
        if n % 2 == 0 {
            return integerReplacement(n/2) + 1
        } else {
            return min(integerReplacement(n-1), integerReplacement(n+1)) + 1
        }
    }
    //412.Fizz Buzz
    func fizzBuzz(_ n: Int) -> [String] {
        var res = [String]()
        for i in 1...n {
            if i % 3 == 0 && i % 5 == 0 {
                res.append("FizzBuzz")
            } else if i % 3 == 0 {
                res.append("Fizz")
            } else if i % 5 == 0 {
                res.append("Buzz")
            } else {
                res.append("\(i)")
            }
        }
        return res
    }
    //441. 排列硬币
    func arrangeCoins(_ n: Int) -> Int {
        let a = Int(sqrt(Double(2 * n)))
        if 2 * n >= a * (a + 1) {
            return a
        }
        return a - 1
    }
    //442.数组中的重复数据
    func findDuplicates(_ nums: [Int]) -> [Int] {
        var res = [Int]()
        var newArr = nums
        for i in 0..<nums.count {
            let index = nums[i] - 1
            if newArr[index] < 0 {
                res.append(index+1)
            } else {
                newArr[index] = -nums[index]
            }
        }
        return res
    }
    //455. 分发饼干
    func findContentChildren(_ g: [Int], _ s: [Int]) -> Int {
        var arrG = g.sorted()
        var arrS = s.sorted()
        var indexG = 0
        var indexS = 0
        var res = 0
        while indexS < s.count && indexG < g.count {
            if arrS[indexS] < arrG[indexG] {
                indexS += 1
            } else {
                indexG += 1
                indexS += 1
                res += 1
            }
        }
        return res
    }
    //458. 可怜的小猪
    func poorPigs(_ buckets: Int, _ minutesToDie: Int, _ minutesToTest: Int) -> Int {
        if buckets == 1 {
            return 0
        }
        let m = minutesToTest / minutesToDie + 1
        var r = 1
        var i = 0
        while(r < buckets) {
            i += 1
            r *= m
        }
        return i
    }
    //462. 最少移动次数使数组元素相等 II
    func minMoves2(_ nums: [Int]) -> Int {
        if nums.count < 2 {
            return 0
        }
        var result = 0
        let newNums = nums.sorted()
        let mid = newNums[newNums.count / 2]
        for i in newNums {
            result += abs(i - mid)
        }
        return result
    }
    //473. 火柴拼正方形
    func makesquare(_ nums: [Int]) -> Bool {
        let sum = nums.reduce(0, +)
        if sum < 4 || nums.count < 4 {
            return false
        }
        var arr = nums.sorted(by: >)
        var sums = [Int](repeating: 0, count: 4)
        let target = sum / 4
        func helper(_ sums: inout [Int],_ nums: [Int],_ pos: Int,_ target: Int) -> Bool {
            if pos >= nums.count {
                return sums[0] == target && sums[1] == target && sums[2] == target && sums[3] == target
            }
            for i in 0..<4 {
                if nums[pos] > target {
                    return false
                }
                if sums[i] + nums[pos] > target {
                    continue
                }
                sums[i] += nums[pos]
                if helper(&sums, nums, pos + 1, target) {
                    return true
                }
                sums[i] -= nums[pos]
            }
            return false
        }
        if sum % 4 == 0 {
            return helper(&sums, arr, 0, target)
        } else {
            return false
        }
    }
    //476.数字的补数
    func findComplement(_ num: Int) -> Int {
        var res = 0
        var i = 0
        var n = num
        while n > 0 {
            res += ((n % 2) ^ 1) << i
            i += 1
            n >>= 1
        }
        return res
    }
    //485. 最大连续1的个数
    func findMaxConsecutiveOnes(_ nums: [Int]) -> Int {
        var result = 0
        var tmp = 0
        for i in nums {
            if i == 0 {
                result = max(result,tmp)
                tmp = 0
            } else {
                tmp += 1
                result = max(result,tmp)
            }
        }
        return result
    }
    //495.提莫攻击
    func findPoisonedDuration(_ timeSeries: [Int], _ duration: Int) -> Int {
        if timeSeries.count == 0 {
            return 0
        }
        var end = 0
        var res = 0
        for i in 0..<timeSeries.count {
            if timeSeries[i] >= end {
                res += duration
            } else {
                res += timeSeries[i] - timeSeries[i-1]
            }
            end = timeSeries[i] + duration
        }
        return res

    }
    //506.相对名次
    func findRelativeRanks(_ nums: [Int]) -> [String] {
        let arr = nums.sorted(by: >)
        var res = [String]()
        var dict = [Int:String]()
        for i in 0..<arr.count {
            if i == 0 {
                dict[arr[i]] = "Gold Medal"
            } else if i == 1 {
                dict[arr[i]] = "Silver Medal"
            } else if i == 2 {
                dict[arr[i]] = "Bronze Medal"
            } else {
                dict[arr[i]] = "\(i+1)"
            }
        }
        for i in nums {
            res.append(dict[i]!)
        }
        return res
    }
    //507.完美数
    func checkPerfectNumber(_ num: Int) -> Bool {
        if num < 6 {
            return false
        }
        var arr = [1]
        let sq = Int(sqrt(Double(num)))
        for i in 2...sq {
            if num % i == 0 {
                arr.append(i)
                arr.append(num / i)
            }
            if sq * sq == num {
                arr.append(sq)
            }
        }
        return arr.reduce(0, +) == num
    }
    //557.反转字符串中的单词 III
    func reverseWords3(_ s: String) -> String {
        var arr = s.components(separatedBy: " ")
        for i in 0..<arr.count {
            arr[i] = String(arr[i].reversed())
        }
        return arr.joined(separator: " ")
    }
    //593.有效正方形
    func validSquare(_ p1: [Int], _ p2: [Int], _ p3: [Int], _ p4: [Int]) -> Bool {
        struct Vector {
            var length: Double
            var point: [Int]
        }
        func pointToVector(_ point1: [Int], _ point2: [Int]) -> Vector {
            let point = [point1.first! - point2.first!,point1.last! - point2.last!]
            let length = sqrt(Double(point.first! * point.first! + point.last! * point.last!))
            let vector = Vector(length: length, point: point)
            return vector
        }
        var vectorArr = [Vector]()
        vectorArr.append(pointToVector(p1, p2))
        vectorArr.append(pointToVector(p1, p3))
        vectorArr.append(pointToVector(p1, p4))
        vectorArr.append(pointToVector(p2, p3))
        vectorArr.append(pointToVector(p2, p4))
        vectorArr.append(pointToVector(p3, p4))
        vectorArr.sort { (s1, s2) -> Bool in
            return s1.length > s2.length
        }
        let v1 = vectorArr[0]
        let v2 = vectorArr[1]
        if v1.length != v2.length || v1.length == 0{
            return false
        }
        if v1.point[0] * v2.point[0] + v1.point[1] * v2.point[1] == 0 {
            return true
        }
        return false
    }
    //663.平方数之和
    func judgeSquareSum(_ c: Int) -> Bool {
        let n = Int(sqrt(Double(c)))
        for i in 0...n {
            if i * i == c {
                return true
            }
            let d = c - i * i
            let p = Int(sqrt(Double(d)))
            if d == p * p {
                return true
            }
        }
        return false
    }
    //643. 子数组最大平均数 I
    func findMaxAverage1(_ nums: [Int], _ k: Int) -> Double {
        var sum = 0
        for i in 0..<k {
            sum += nums[i]
        }
        var res = sum
        for i in 1..<nums.count - k + 1 {
            sum -= nums[i-1]
            sum += nums[i-1+k]
            res = max(res,sum)
        }
        return Double(res)/Double(k)
    }
    //697. 数组的度
    func findShortestSubArray(_ nums: [Int]) -> Int {
        var dict = [Int: Int]()
        for i in nums {
            if let n = dict[i] {
                dict[i] = n + 1
            } else {
                dict[i] = 1
            }
        }
        var arr = [Int]()
        var count = 0
        for (key,value) in dict {
            if value > count {
                count = value
                arr.removeAll()
                arr.append(key)
            }
            if value == count {
                arr.append(key)
            }
        }
        var res = nums.count
        for i in arr {
            var left = 0
            var right = nums.count - 1
            while left < right {
                if nums[left] != i {
                    left += 1
                }
                if nums[right] != i {
                    right -= 1
                }
                if nums[left] == i && nums[right] == i {
                    res = min(res, right - left + 1)
                    break
                }
            }
        }
        return res
    }
    //713. 乘积小于K的子数组
    func numSubarrayProductLessThanK(_ nums: [Int], _ k: Int) -> Int {
        if ( k <= 1) {
            return 0
        }
        let n = nums.count
        var p = 1
        var i = 0
        var total = 0
        for j in 0..<n {
            p *= nums[j]
            while (p >= k){
                p /= nums[i]
                i += 1
            }
            total += (j - i + 1)
        }
        return total
    }
    //724. 寻找数组的中心索引
    func pivotIndex(_ nums: [Int]) -> Int {
        if nums.count < 3 {
            return -1
        }
        var leftArr = [Int](repeating:0, count: nums.count)
        leftArr[0] = nums[0]
        var rightArr = [Int](repeating: 0, count: nums.count)
        rightArr[nums.count-1] = nums[nums.count-1]
        for i in 1..<nums.count {
            leftArr[i] = leftArr[i-1] + nums[i]
            rightArr[nums.count-1-i] = rightArr[nums.count-i] + nums[nums.count-1-i]
        }
        for i in 0..<nums.count {
            if leftArr[i] == rightArr[i] {
                return i
            }
        }
        return -1
    }
    //728.自除数
    func selfDividingNumbers(_ left: Int, _ right: Int) -> [Int] {
        func isTrue(_ num: Int) -> Bool {
            var n = num
            while n > 0 {
                let m = n % 10
                if m == 0 {
                    return false
                }
                if num % m != 0 {
                    return false
                }
                n /= 10
            }
            return true
        }
        var res = [Int]()
        for i in left...right {
            if isTrue(i) {
                res.append(i)
            }
        }
        return res
    }
    //738.单调递增的数字
    func monotoneIncreasingDigits(_ N: Int) -> Int {
        var arr = [Int]()
        var n = N
        while n > 0 {
            arr.append(n % 10)
            n /= 10
        }
        let len = arr.count
        var j = 0
        for i in 1..<len {
            if arr[i] > arr[i-1] {
                arr[i] -= 1
                j = i
            }
        }
        for i in 0..<j {
            arr[i] = 9
        }
        var res = 0
        for i in arr.reversed()  {
            res = res * 10 + i
        }
        return res
    }
    //739.每日温度
    func dailyTemperatures(_ temperatures: [Int]) -> [Int] {
        var res = [Int]()
        var stack = [Int]()
        for i in (0...temperatures.count - 1).reversed() {
            while stack.count > 0 && temperatures[stack.last!] <= temperatures[i] {
                stack.removeLast()
            }
            let day = stack.count > 0 ? stack.last! - i : 0
            res.insert(day, at: 0)
            stack.append(i)
        }
        return res
    }
    //754. 到达终点数字
    func reachNumber(_ target: Int) -> Int {
        var n = Int(abs(Int32(target)))
        var i = 1
        while true {
            n -= 1
            if n == 0 || (n < 0 && n % 2 == 0){
                return i
            }
            i += 1
        }
    }
    //766. 托普利茨矩阵
    func isToeplitzMatrix(_ matrix: [[Int]]) -> Bool {
        for i in 0..<matrix.count {
            for j in 0..<matrix[0].count {
                if matrix[i][j] != matrix[i+1][j+1] {
                    return false
                }
            }
        }
        return true
    }
    //799.香槟塔
    func champagneTower(_ poured: Int, _ query_row: Int, _ query_glass: Int) -> Double {
        let maxCount = max(query_row,query_glass)
        var array = [[Double]](repeating: [Double](repeating: 0, count: maxCount+1), count: maxCount+1)
        array[0][0] = Double(poured)
        for i in 0..<maxCount {
            for j in 0..<maxCount {
                if (array[i][j] - 1) / 2 > 0 {
                    array[i+1][j] += (array[i][j] - 1)/2
                    array[i+1][j+1] += (array[i][j] - 1)/2
                }
            }
        }
        let res = array[query_row][query_glass]
        if res < 0 {
            return 0
        }
        if res > 1 {
            return 1
        }
        return res
    }
    //804. 唯一摩尔斯密码词 *******swift中转换ASCII码有问题 故用的dict 没用数组
    func uniqueMorseRepresentations(_ words: [String]) -> Int {
        let dict = ["a":".-","b":"-...","c":"-.-.","d":"-..","e":".","f":"..-.","g":"--.","h":"....","i":"..","j":".---","k":"-.-","l":".-..","m":"--","n":"-.","o":"---","p":".--.","q":"--.-","r":".-.","s":"...","t":"-","u":"..-","v":"...-","w":".--","x":"-..-","y":"-.--","z":"--.."]
        var set = Set<String>()
        for s in words {
            var string = ""
            for c in s {
                string += dict[String(c)]!
            }
            set.insert(string)
        }
        return set.count
    }
    //811. 子域名访问计数
    func subdomainVisits(_ cpdomains: [String]) -> [String] {
        var domainDict = [String: Int]()
        func calDomain(_ num: Int,_ string: String,_ dict: inout [String: Int]) {
            var arr = string.components(separatedBy: ".")
            if arr.count > 1 {
                calDomain(num, String(string.dropFirst(arr[0].count + 1)), &dict)
            }
            if let count = dict[string] {
                dict[string] = num + count
            } else {
                dict[string] = num
            }

        }
        for domain in cpdomains {
            let arr = domain.components(separatedBy: " ")
            calDomain(Int(arr[0])!, arr[1], &domainDict)
        }
        return domainDict.map { (key,value) -> String in
            return "\(value) \(key)"
        }
    }
    //824.山羊拉丁文
    func toGoatLatin(_ S: String) -> String {
        func isAEIOU(_ string: String) -> Bool {
            let newStr = string.lowercased()
            if newStr.hasPrefix("a") || newStr.hasPrefix("e") || newStr.hasPrefix("i") || newStr.hasPrefix("o") || newStr.hasPrefix("u") {
                return true
            }
            return false
        }
        func moveToLast(_ s: String) -> String {
            var newStr = s
            let c = s.first!
            newStr = String(newStr.dropFirst())
            newStr.append(c)
            return newStr + "ma"
        }
        func addAWithIndex(_ s: String,_ index: Int) -> String {
            var newStr = s
            newStr += "a"
            for _ in 0..<index {
                newStr += "a"
            }
            return newStr
        }
        var arr = S.components(separatedBy: " ")
        for (i,str) in arr.enumerated() {
            if isAEIOU(str) {
                arr[i] = arr[i] + "ma"
            } else {
                arr[i] = moveToLast(arr[i])
            }
            arr[i] = addAWithIndex(arr[i], i)
        }
        return arr.joined(separator: " ")
    }
    //829.连续整数求和
    func consecutiveNumbersSum(_ N: Int) -> Int {
        if N < 3 {
            return 1
        }
        let limit = Int(0.5 + sqrt(Double(2*N) + 0.25))
        var res = 0
        for i in 1...limit {
            if Double(N) / Double(i) > Double(i / 2) {
                if i % 2 == 0 {
                    if N % i == i / 2 {
                        res += 1
                    }
                } else {
                    if N % i == 0 {
                        res += 1
                    }
                }
            }
        }
        return res
    }
    //846. 一手顺子 题目好像没读懂。。。。 超时。。。需优化
    func isNStraightHand(_ hand: [Int], _ W: Int) -> Bool {
        if hand.count % W != 0 {
            return false
        }
        var arr = hand.sorted()
        var matrix = [[Int]]()
        matrix.append([arr[0]])
        for i in 1..<arr.count {
            for j in 0..<matrix.count {
                if arr[i] == matrix[j].last! + 1 && matrix[j].count < W {
                    matrix[j].append(arr[i])
                    break
                }
                if j == matrix.count - 1{
                    let a = [arr[i]]
                    matrix.append(a)
                    break
                }
            }
        }
        return matrix.reduce(true, {
            $0 && $1.count == W
        })
    }
    //860.柠檬水找零
    func lemonadeChange(_ bills: [Int]) -> Bool {
        var arr5 = [Int]()
        var arr10 = [Int]()
        for i in bills {
            if i == 5 {
                arr5.append(i)
            }
            if i == 10 {
                if arr5.count < 1 {
                    return false
                }
                arr5.removeLast()
                arr10.append(i)
            }
            if i == 20 {
                if arr5.count > 0 && arr10.count > 0 {
                    arr5.removeLast()
                    arr10.removeLast()
                } else if arr5.count > 3 && arr10.count < 1 {
                    arr5.removeLast()
                    arr5.removeLast()
                    arr5.removeLast()
                } else {
                    return false
                }
            }
        }
        return true
    }
    //862.最短子数组和至少为k ****报错****
    func shortestSubarray(_ A: [Int], _ K: Int) -> Int {
        guard A.count > 0 else {
            return -1
        }
        var left = 0
        var right = 0
        var sum = 0
        var res = 0
        while right < A.count {
            sum += A[right]
            if sum >= K {
                if res == 0 {
                    res = right - left + 1
                } else {
                    res = min(res, right - left + 1)
                }
                if res == 1 {
                    return res
                }
                while left < right {
                    sum -= A[left]
                    left += 1
                    if sum >= K {
                        res = min(res, right - left + 1)
                    } else {
                        if A[left] > 0 {
                            right += 1
                            break
                        }
                    }
                }
            } else {
                right += 1
                if right == A.count {
                    if res == 0 {
                        return -1
                    } else {
                        return res
                    }
                }
            }
        }
        return res
    }

}
