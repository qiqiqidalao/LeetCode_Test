using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeetCode_Test
{
    class Program
    {
        public class TreeNode
        {
            public int val;
            public TreeNode left;
            public TreeNode right;
            public TreeNode(int x) { val = x; }
        }

        public class ListNode
        {
            public int val;
            public ListNode next;
            public ListNode(int x) { val = x; }
        }

        static void Main(string[] args)
        {
            //string str = "bacbababadababacambabacaddababacasdsd";
            //string ctr = "ababaca";
            //List<int> res = KMP(str, ctr);
            //foreach (int temp in res)
            //    Console.WriteLine("下标：" + temp + "存在匹配字符串");
            Console.WriteLine(InfixExpToPostfixExp("(a*(b+c/d-e)+f)"));
            Console.Read();
        }

        #region LeetCode_98
        public bool IsValidBST(TreeNode root)
        {
            List<int> num = new List<int>();
            solve(root, num);
            for(int i = 0; i < num.Count - 1; i++)
            {
                if (num[i] >= num[i + 1])
                    return false;
            }
            return true;
        }
        public void solve(TreeNode root,List<int> num)
        {
            if (root != null)
            {
                solve(root.left, num);
                num.Add(root.val);
                solve(root.right, num);
            }
        }
        #endregion

        #region LeetCode_101
        //别人的写法
        public bool IsSymmetric(TreeNode root)
        {
            if (root == null)
                return true;

            return IsSymmetric(root.left, root.right);

        }
        public bool IsSymmetric(TreeNode left, TreeNode right)
        {
            if (left == null && right == null)
                return true;

            if (left != null && right != null)
            {
                if (left.val != right.val)
                    return false;

                return IsSymmetric(left.right, right.left) && IsSymmetric(left.left, right.right);
            }

            return false;
        }
        /*LeetCode_101 自己写的
        public bool IsSymmetric(TreeNode root)
        {
            List<int> f_num = new List<int>();
            List<int> re_f_num = new List<int>();
            solve_IsSymmertric(root, f_num);
            solve_Re_IsSymmertric(root, re_f_num);
            for (int i = 0; i < f_num.Count; i++)
                Console.Write(f_num[i]);
            Console.WriteLine();
            for (int i = 0; i < re_f_num.Count; i++)
                Console.Write(re_f_num[i]);
            if (f_num.Count != re_f_num.Count) return false;
            for (int i = 0; i < f_num.Count; i++)
                if (f_num[i] != re_f_num[i]) return false;
            return true;
        }
        public void solve_IsSymmertric(TreeNode root, List<int> num)
        {
            if (root != null)
            {
                num.Add(root.val);
                solve_IsSymmertric(root.left, num);
                solve_IsSymmertric(root.right, num);
            }
            else
                num.Add(-1);
        }
        public void solve_Re_IsSymmertric(TreeNode root, List<int> num)
        {
            if (root != null)
            {
                num.Add(root.val);
                solve_Re_IsSymmertric(root.right, num);
                solve_Re_IsSymmertric(root.left, num);
            }
            else
                num.Add(-1);
        }*/
        #endregion

        #region LeetCode_108
        //别人的写法
        public TreeNode SortedArrayToBST(int[] nums)
        {
            if (nums.Length == 0) return null;
            return solve_SortedArrayToBST(0, nums.Length - 1, nums);
        }
        public TreeNode solve_SortedArrayToBST(int start, int end, int[] nums)
        {
            if (start > end) return null;
            int mid = (start + end) / 2;
            TreeNode root = new TreeNode(nums[mid]);
            root.left = solve_SortedArrayToBST(start, mid - 1, nums);
            root.right = solve_SortedArrayToBST(mid + 1, end, nums);
            return root;
        }
        #endregion

        #region LeetCode_155
        public class MinStack
        {
            Stack<int> min;
            Stack<int> all;
            /** initialize your data structure here. */
            public MinStack()
            {
                min = new Stack<int>();
                all = new Stack<int>();
            }

            public void Push(int x)
            {
                if (all.Count == 0)
                {
                    all.Push(x);
                    min.Push(x);
                }
                else
                {
                    if (x < min.Peek()) min.Push(x);
                    else min.Push(min.Peek());
                    all.Push(x);
                }
            }

            public void Pop()
            {
                all.Pop();
                min.Pop();
            }

            public int Top()
            {
                return all.Peek();
            }

            public int GetMin()
            {
                return min.Peek();
            }
        }
        #endregion

        #region LeetCode_384
        //思路:遍历数组每个位置，每次都从该位置后面随机生成一个坐标位置，然后交换当前遍历位置和随机生成的坐标位置的数字，
        //这样如果数组有n个数字，那么我们也随机交换了n组位置，从而达到了洗牌的目的，
        //注意i + rand() % (res.size() - i)不能写成rand() % res.size()
        public class Solution
        {
            int[] nums, re;
            Random random;
            public Solution(int[] nums)
            {
                this.nums = nums;
                re = nums;
                random = new Random();
            }

            /** Resets the array to its original configuration and return it. */
            public int[] Reset()
            {
                return re;
            }

            /** Returns a random shuffling of the array. */
            public int[] Shuffle()
            {
                nums = (int[])re.Clone();
                for(int i = 0; i < nums.Length; i++)
                {
                    int temp = i + random.Next() % (nums.Length - i);
                    int t = nums[temp];
                    nums[temp] = nums[i];
                    nums[i] = t;
                }
                return nums;
            }

        }
        #endregion

        #region LeetCode_49
        //两个for循环可以合并成一个
        public IList<IList<string>> GroupAnagrams(string[] strs)
        {
            List<string> sort_strs = new List<string>();
            for (int i = 0; i < strs.Length; i++)
                sort_strs.Add(String.Join("", strs[i].ToCharArray().OrderBy(o => o)));
            Dictionary<string, IList<string>> res = new Dictionary<string, IList<string>>();
            for(int i = 0; i < sort_strs.Count; i++)
            {
                if (res.ContainsKey(sort_strs[i]))
                    res[sort_strs[i]].Add(strs[i]);
                else
                    res.Add(sort_strs[i], new List<string>() { strs[i] });
            }
            return res.Values.ToList();
        }
        #endregion

        #region LeetCode_5
        // 思路：中心扩展算法
        // 1，自我比较，然后再检查自己两边的元素，从而得到以i为中心点的回文
        // 2，比较i和i+1，然后再检查i和i+1两边的元素，从而得到以i和i+1为中心点的回文
        // 算法中记录回文的开始和最大长度，这样就不必存储回文信息
        // 扩展方法中返回回文的长度，然后比较1、2中最长的回文，最后主方法根据长度去计算回文的开始与结束点
        /*//存储回文信息的方法
        public string LongestPalindrome(string s)
        {
            int i = 0, j = 0, right = 0;
            string res = "";
            while (i < s.Length)
            {
                string s1 = "", s2 = "";
                s1 = Solve_LongestPalindrome(s, i, i);
                if (res.Length < s1.Length) res = s1;
                if (i != s.Length - 1)
                {
                    s2 = Solve_LongestPalindrome(s, i, i + 1);
                    if (res.Length < s2.Length) res = s2;
                }
                i++;
            }
            return res;
        }

        public string Solve_LongestPalindrome(string s, int left, int right)
        {
            while (left >= 0 && right <= s.Length - 1 && s[left] == s[right])
            {
                left--; right++;
            }
            return s.Substring(left + 1, right - left - 1);
        }*/
        public string LongestPalindrome(string s)
        {
            int max = 0, left = 0;
            for(int i = 0; i < s.Length; i++)
            {
                //检查以i为中心点的回文长度
                int temp1 = Solve_LongestPalindrome(s, i, i);
                //检查以i + 1为中心点的回文长度
                int temp2 = Solve_LongestPalindrome(s, i, i + 1);
                int count = Math.Max(temp1, temp2);
                //根据回文长度计算开始和结束点
                if (count > max)
                {
                    max = count;
                    left = i - (max - 1) / 2;
                }
            }
            return s.Substring(left, max);
        }
        public int Solve_LongestPalindrome(string s,int left,int right)
        {
            while (left >= 0 && right <= s.Length - 1 && s[left] == s[right])
            {
                left--; right++;
            }
            return right - left - 1;
        }

        //LeetCode_334
        public bool IncreasingTriplet(int[] nums)
        {
            if (nums.Length < 3) return false;
            int min = Int32.MaxValue, mid = Int32.MaxValue;
            //先找最小的 找完最小的找次小的 次小的找到后就不用管最小的了
            //因为已知了前面有两个递增的 只需判断是否有比次小的大的就行
            for (int i = 0; i < nums.Length; i++)
            {
                if (nums[i] < min) min = nums[i];
                else if (nums[i] > min && mid > nums[i]) mid = nums[i];
                else if (nums[i] > mid) return true;
            }
            return false;
        }
        #endregion

        #region LeetCode_674
        public int FindLengthOfLCIS(int[] nums)
        {
            if (nums.Length == 0) return 0;
            int max = 0, temp = 1;
            for(int i = 1; i < nums.Length; i++)
            {
                if (nums[i] > nums[i - 1])
                    temp++;
                else
                {
                    if (temp > max) max = temp;
                    temp = 1;
                }
            }
            return temp > max ? temp : max;
        }
        #endregion

        #region LeetCode_673
        public int FindNumberOfLIS(int[] nums)
        {
            if (nums.Length == 0) return 0;
            //dp存储子序列的长度 count存储该长度下有几个
            int[] dp = new int[nums.Length], count = new int[nums.Length];
            int max_Length = 1, res = 0;
            //初始化都为1
            for (int i = 0; i < count.Length; i++)
            {
                count[i] = 1;
                dp[i] = 1;
            }
            for (int i = 1; i < nums.Length; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    //dp[j] + 1大于dp[i] 说明是新的序列
                    if (nums[j] < nums[i] && dp[j] + 1 > dp[i])
                    {
                        dp[i] = dp[j] + 1;
                        count[i] = count[j];
                    }
                    //dp[j] + 1等于dp[i] 说明有长度相同的序列
                    else if (nums[j] < nums[i] && dp[j] + 1 == dp[i])
                    {
                        count[i] += count[j];
                    }
                }
                max_Length = Math.Max(dp[i], max_Length);
            }
            //将所有最大长度的序列个数求和
            for (int i = 0; i < dp.Length; i++)
            {
                if (dp[i] == max_Length)
                    res += count[i];
            }
            return res;
        }
        #endregion

        #region LeetCode_119
        public IList<int> GetRow(int rowIndex)
        {
            if (rowIndex == 0) return new List<int>() { 1 };
            int[][] array = new int[rowIndex + 1][];
            for (int i = 0; i <= rowIndex; i++)
            {
                array[i] = new int[i + 1];
                array[i][0] = array[i][i] = 1;
            }
            for (int i = 2; i <= rowIndex; i++)
            {
                for (int j = 1; j < array[i].Length - 1; j++)
                    array[i][j] = array[i - 1][j - 1] + array[i - 1][j];
            }
            return array[rowIndex].ToList();
        }
        #endregion

        #region LeetCode_448
        public IList<int> FindDisappearedNumbers(int[] nums)
        {
            IList<int> res = new List<int>();
            int index = 0;
            for (int i = 0; i < nums.Length; i++)
            {
                index = Math.Abs(nums[i]) - 1;
                if (nums[index] > 0)
                    nums[index] = -nums[index];
            }

            for(int i = 0; i < nums.Length; i++)
            {
                if (nums[i] > 0)
                    res.Add(i + 1);
            }
            return res;
        }
        #endregion

        #region LeetCode_628
        public int MaximumProduct(int[] nums)
        {
            int first = Int32.MinValue, second = Int32.MinValue, third = Int32.MinValue;
            int f_first = 0, f_second = 0;//记录最小两个数
            for (int i = 0; i < nums.Length; i++)
            {
                if (nums[i] >= first)
                {
                    third = second;
                    second = first;
                    first = nums[i];
                }
                else if (nums[i] >= second)
                {
                    third = second;
                    second = nums[i];
                }
                else if (nums[i] > third)
                    third = nums[i];
                if (nums[i] <= f_first)
                {
                    f_second = f_first;
                    f_first = nums[i];
                }
                else if (nums[i] < f_second)
                    f_second = nums[i];
            }
            int a = first * second * third;
            int b = f_first * f_second * first;
            return a > b ? a : b;
        }
        #endregion

        #region LeetCode_64
        public int MinPathSum(int[,] grid)
        {
            //自己写的
            int[,] dp = new int[grid.GetLength(0), grid.GetLength(1)];
            dp[0, 0] = grid[0, 0];
            for (int i = 1; i < grid.GetLength(0); i++)
                dp[i, 0] = grid[i, 0] + dp[i - 1, 0];
            for (int j = 1; j < grid.GetLength(1); j++)
                dp[0, j] = grid[0, j] + dp[0, j - 1];
            for(int i = 1; i < grid.GetLength(0); i++)
            {
                for(int j = 1; j < grid.GetLength(1); j++)
                {
                    dp[i, j] = Math.Min(dp[i - 1, j], dp[i, j - 1]) + grid[i, j];
                }
            }
            return dp[dp.GetLength(0) - 1, dp.GetLength(1) - 1];
            //别人一个循环的版本 思路一样
            //if (grid == null || grid.Length == 0)
            //    return 0;
            //int rows = grid.GetLength(0);
            //int cols = grid.GetLength(1);
            //int[,] dp = new int[rows, cols];
            //dp[0, 0] = grid[0, 0];
            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {
            //        if (i == 0 || j == 0)
            //        {
            //            if (i == 0 && j != 0)
            //            {
            //                dp[i, j] = dp[i, j - 1] + grid[i, j];
            //            }
            //            else if (i != 0 && j == 0)
            //            {
            //                dp[i, j] = dp[i - 1, j] + grid[i, j];
            //            }
            //        }
            //        else
            //        {
            //            int min = dp[i - 1, j] < dp[i, j - 1] ? dp[i - 1, j] : dp[i, j - 1];
            //            dp[i, j] = min + grid[i, j];
            //        }
            //    }
            //}
            //return dp[rows - 1, cols - 1];
        }
        #endregion

        #region LeetCode_896
        public bool IsMonotonic(int[] A)
        {
            if (A.Length == 0) return false;
            if (A.Length == 1) return true;
            bool a = false, b = false;//a记录递增，b记录递减
            for (int i = 1; i < A.Length; i++)
            {
                if (A[i] > A[i - 1])
                    a = true;
                else if (A[i] < A[i - 1])
                    b = true;
                else
                    continue;
            }
            return a && b ? false : true;
        }
        #endregion

        #region LeetCode_343
        //思路：
        //使用dp[i]表示正整数i最大的乘积
        //dp[i] = Max(Max(dp[i-1] * 1,1 * (i - 1)),Max(dp[i-2] * 2,2 * (i - 2)))
        //dp[0] = 0;dp[1] = 1;
        //dp[2] = 1 * 1;dp[3] = 1 * 2;
        //dp[4] = Max(Max(dp[3] * 1,1 * 3),Max(dp[2] * 2,2 * 2),Max(dp[1] * 3,3 * 1))

        //优化状态转移方程
        //这些正整数拆分最终总会拆分为2,3和少数的1，所以可以优化为
        //d[i] = Max(Max(dp[i-2] * 2,2 * (i-2)),Max(dp[i-3] * 3,3 * (i-3)));
        public int IntegerBreak(int n)
        {
            int[] dp = new int[n + 1];
            dp[0] = 0;dp[1] = 1;dp[2] = 1;dp[3] = 2;
            for(int i = 4; i <= n; i++)
            {
                //优化的写法
                //int temp1, temp2;
                //temp1 = Math.Max(dp[i - 2] * 2, 2 * (i - 2));
                //temp2 = Math.Max(dp[i - 3] * 3, 3 * (i - 3));
                //dp[i] = temp1 > temp2 ? temp1 : temp2;
                int max = 0;
                for(int j = i - 1; j >= 1; j--)
                {
                    int temp = Math.Max(dp[j] * (i - j), (i - j) * j);
                    if (temp > max) max = temp;
                }
                dp[i] = max;
            }
            return dp[n];
        }
        #endregion

        #region LeetCode_152
        //思路:
        //访问到每个点的时候，以该点为子序列的末尾的乘积，
        //要么是该点本身，要么是该点乘以以前一点为末尾的序列，
        //注意乘积负负得正，故需要记录前面的最大最小值。
        //例如    2   3   -2    4   -5    -30
        //Max     2   6   -2    4   240   600
        //Min     2   2   -12  -48  -20   -7200
        //Res     2   6    6    6    240   600
        public int MaxProduct(int[] nums)
        {
            int Max = nums[0];//记录最大值
            int Min = nums[0];//记录最小值
            int res = nums[0];
            for (int i = 1; i < nums.Length; i++)
            {
                int tempMax = Max;
                int tempMin = Min;
                Max = Math.Max(nums[i], Math.Max(nums[i] * tempMax, nums[i] * tempMin));
                Min = Math.Min(nums[i], Math.Min(nums[i] * tempMax, nums[i] * tempMin));
                res = Max > res ? Max : res;
            }
            return res;
        }
        #endregion

        #region LeetCode_32
        public int LongestValidParentheses(string s)
        {
            int start = 0, max = 0;
            //记录左括号的下标
            Stack<int> left = new Stack<int>();
            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] == '(')
                {
                    left.Push(i);
                }
                else
                {
                    //没有左括号 出现右括号时
                    if (left.Count == 0) start = i + 1;
                    else
                    {
                        left.Pop();
                        //如果栈为空 则判断现在的位置到记录的启始位置+1是否大于max +1是因为记录的匹配的左括号位置
                        if (left.Count == 0) max = Math.Max(max, i - start + 1);
                        //否则判断现在的位置距离上一个左括号的位置(即一个完整括号的长度,所以不用+1)是否大于max
                        else max = Math.Max(max, i - left.Peek());
                    }
                }
            }
            return max;
        }
        #endregion

        #region KMP测试 详见https://blog.csdn.net/starstar1992/article/details/54913261
        //最长前缀：从第一个字符开始，不包括最后一个字符 例：aaaa的最长前缀为aaa
        public static void Cal_NextArray(string str, int[] next)
        {
            next[0] = -1;
            int k = -1;
            for(int i = 1; i < str.Length; i++)
            {
                //如果下一个不同，那么k就变成next[k]，注意next[k]是小于k的，无论k取任何值。
                while (k > 0 && str[k + 1] != str[i])
                    k = next[k];//往前回溯
                if (str[k + 1] == str[i])
                    k = k + 1;
                next[i] = k;
            }
        }

        public static List<int> KMP(string str,string ctr)
        {
            List<int> res = new List<int>();
            int[] next = new int[ctr.Length];//不是ctr.Length - 1
            Cal_NextArray(ctr, next);
            int k = -1;
            for(int i = 0; i < str.Length; i++)
            {
                //ptr和str不匹配，且k>-1（表示ptr和str有部分匹配）
                while (k > -1 && ctr[k + 1] != str[i])
                    k = next[k];//往前回溯
                if (ctr[k + 1] == str[i])
                    k = k + 1;
                if (k == ctr.Length - 1)
                {
                    res.Add(i - ctr.Length + 1);
                    //重新初始化，寻找下一个
                    k = -1;
                    i = i - ctr.Length + 1;
                }
            }
            return res;
        }
        #endregion

        #region LeetCode_258
        public int AddDigits(int num)
        {
            //不用循环或者递归 O(1)的时间复杂度
            //return num % 9 == 0 ? (num == 0 ? 0 : 9) : num % 9;
            while (num >= 10)
            {
                int sum = 0;
                while (num >= 10)
                {
                    sum = sum + num % 10;
                    num = num / 10;
                }
                sum += num;
                num = sum;
            }
            return num;
        }
        #endregion

        #region LeetCoed_441
        public int ArrangeCoins(int n)
        {
            int res = 1;
            while (n >= res)
            {
                n -= res;
                res++;
            }
            return res - 1;
        }
        #endregion

        #region LeetCode_561
        public int ArrayPairSum(int[] nums)
        {
            Array.Sort(nums);
            int res = 0;
            for (int i = 0; i < nums.Length; i = i + 2)
                res += nums[i];
            return res;   
        }
        #endregion

        #region LeetCode_520
        public bool DetectCapitalUse(string word)
        {
            //别人的写法
            //return word == word.ToLower() ||
            //word == word.ToUpper() ||
            //(word[0].ToString() == word[0].ToString().ToUpper() &&
            //word.Substring(1) == word.Substring(1).ToLower());
            if (word.Length <= 1) return true;
            bool flag = word[1] >= 'a' && word[1] <= 'z' ? false : true;
            for (int i = 2; i < word.Length; i++)
            {
                if (flag)
                {
                    if (word[i] >= 'a' && word[i] <= 'z')
                        return false;
                }
                else
                {
                    if (word[i] >= 'A' && word[i] <= 'Z')
                        return false;
                }
            }
            return flag ? word[0] >= 'a' && word[0] <= 'z' ? false : true : true;
        }
        #endregion

        #region LeetCode_704
        public int Search(int[] nums, int target)
        {
            int start = 0, end = nums.Length - 1;
            while (start <= end)
            {
                int mid = (start + end) / 2;
                if (nums[mid] == target) return mid;
                else if (nums[mid] > target) end = mid - 1;
                else start = mid + 1;
            }
            return -1;
        }
        #endregion

        #region 中序表达式转后序表达式
        public static string InfixExpToPostfixExp(string str)
        {
            //假设输入完全合法 数字用小写字母代表
            str = str.ToLower();
            StringBuilder res = new StringBuilder("");
            Stack<char> temp = new Stack<char>();
            Dictionary<char, int> find = new Dictionary<char, int>();
            find.Add('(', -1); find.Add('+', 0);find.Add('-', 0);find.Add('*', 1);find.Add('/', 1);
            for(int i = 0; i < str.Length; i++)
            {
                if (str[i] >= 'a' && str[i] <= 'z') res.Append(str[i]);
                else if (str[i] == '+' || str[i] == '-' || str[i] == '/' || str[i] == '*')
                {
                    if (temp.Count == 0)
                        temp.Push(str[i]);
                    else
                    {
                        while (temp.Count != 0)
                        {
                            int p = find[temp.Peek()], q = find[str[i]];
                            if (p < q)
                            {
                                temp.Push(str[i]);
                                break;
                            }
                            else
                                res.Append(temp.Pop());
                        }
                        if (temp.Count == 0) temp.Push(str[i]);
                    }
                }
                else
                {
                    if (str[i] == '(') temp.Push(str[i]);
                    else
                    {
                        while (temp.Count != 0 && temp.Peek() != '(')
                        {
                            res.Append(temp.Pop());
                        }
                        if (temp.Count != 0) temp.Pop();
                    }
                }
            }
            while (temp.Count != 0) res.Append(temp.Pop());
            return res.ToString();
        }
        #endregion

        #region LeetCode_51 LeetCode_52同理只要返回解决方案个数
        public IList<IList<string>> SolveNQueens(int n)
        {
            IList<IList<string>> res = new List<IList<string>>();
            //N皇后棋盘 下标表示行，值代表列，不用0行0列
            int[] checkerboard = new int[n + 1];
            NQueens(res, checkerboard, 1);//回溯 从第一行开始
            return res;
        }

        public void NQueens(IList<IList<string>> res, int[] cb, int current)
        {
            //如果现在的行号等于输入的n 说明排列完成
            if (current == cb.Length)
            {
                List<string> r = new List<string>();
                for (int i = 1; i < cb.Length; i++)
                {
                    StringBuilder temp = new StringBuilder("");
                    //cb.Length比n大1 故构造一行的循环次数为cb.Length - 1
                    //当然也可以用for (int j = 1; j< cb.Length; j++) 下标不同而已
                    for (int j = 0; j < cb.Length - 1; j++)
                    {
                        if (cb[i] - 1 == j) temp.Append('Q');
                        else temp.Append('.');
                    }
                    r.Add(temp.ToString());
                }
                res.Add(r);
            }
            else
            {
                //这个循环并不是针对行 而是针对列
                //即设置当前皇后的列cb[current] = i;
                for (int i = 1; i < cb.Length; i++)
                {
                    cb[current] = i;
                    //判断是否有同列同对角线，不用考虑同行
                    if (check_NQueens(cb, current))
                    {
                        //如果当前位置不冲突，进行下一个皇后的放置
                        NQueens(res, cb, current + 1);
                    }
                    //否则换一个位置 再判断
                }
            }
            return;
        }

        public bool check_NQueens(int[] cb, int current)
        {
            //从第二个皇后开始和之前的皇后位置进行判断，直到当前皇后放的位置
            for (int i = 2; i <= current && i < cb.Length; i++)
            {
                for (int j = 1; j < i; j++)
                {
                    //{i,cb[i]} {j,cb[j]}表示两个皇后
                    //cb[i]==cb[j]表示同列
                    //Math.Abs(cb[i] - cb[j]) == i - j表示同对角线(两条)
                    //回溯条件保证了i与j不可能相等，所以不考虑行相等
                    if (cb[i] == cb[j] || (Math.Abs(cb[i] - cb[j]) == i - j))
                    {
                        return false;
                    }
                }
            }
            return true;
        }
        #endregion

        #region LeetCode_203
        public ListNode RemoveElements(ListNode head, int val)
        {
            if (head == null) return null;
            while (head != null && head.val == val) head = head.next;
            if (head == null) return null;
            ListNode note = head,temp1 = head, temp2 = head.next;
            while (temp2 != null)
            {
                if (temp2.val == val)
                {
                    temp1.next = temp2.next;
                    temp2 = temp2.next;
                }
                else
                {
                    temp1 = temp1.next;
                    temp2 = temp2.next;
                }
            }
            return note;
        }
        #endregion

        #region LeetCode_144
        public IList<int> PreorderTraversal(TreeNode root)
        {
            IList<int> res = new List<int>();
            //递归实现
            //solve_PreorderTraversal(root, res);
            //return res;
            //非递归实现
            Stack<TreeNode> temp = new Stack<TreeNode>();
            while (root != null || temp.Count != 0)
            {
                while (root != null)//while改成if同样可以
                {
                    res.Add(root.val);
                    temp.Push(root);
                    root = root.left;
                }
                if (temp.Count != 0)
                {
                    root = temp.Pop();
                    root = root.right;
                }
            }
            return res;
        }

        public void solve_PreorderTraversal(TreeNode root,IList<int> res)
        {
            if (root != null)
            {
                res.Add(root.val);
                solve_PreorderTraversal(root.left, res);
                solve_PreorderTraversal(root.right, res);
            }
        }
        #endregion

        #region LeetCode_728
        public IList<int> SelfDividingNumbers(int left, int right)
        {
            IList<int> res = new List<int>();
            for (int i = left; i < 10; i++)
                res.Add(i);
            for (int i = left < 11 ? 11 : left; i <= right; i++)
            {
                int temp = i;
                bool flag = true;
                while (temp != 0)
                {
                    if (temp % 10 == 0)
                    {
                        flag = false;
                        break;
                    }
                    if (i % (temp % 10) != 0)
                    {
                        flag = false;
                        break;
                    }
                    temp = temp / 10;
                }
                if (flag) res.Add(i);
            }
            return res;
        }
        #endregion

        #region LeetCode_459
        public bool RepeatedSubstringPattern(string s)
        {
            //没看懂别人写的
            //var i = 1;
            //var j = 0;
            //var n = s.Length;
            //var dp = new int[n + 1];
            //while (i < n)
            //{
            //    if (s[i] == s[j]) dp[++i] = ++j;
            //    else if (j == 0) ++i;
            //    else j = dp[j];
            //}
            //return dp[n] >= 1 && (dp[n] % (n - dp[n]) == 0);
            
            //自己写的
            if (s.Length <= 1) return false;
            for (int cut = 1; cut <= s.Length / 2; cut++)
            {
                if (s.Length % cut == 0 && checkRepeatedStr(s, cut))
                    return true;
            }
            return false;
        }

        public bool checkRepeatedStr(string s, int cut)
        {
            for(int i = 0; i < cut; i++)
            {
                for (int j = i + cut; j < s.Length; j = j + cut)
                {
                    if (s[i] != s[j]) return false;
                }
            }
            return true;
        }
        #endregion

        #region LeetCode_404
        public int SumOfLeftLeaves(TreeNode root)
        {
            //另一种写法
            //var sum = 0;
            //PreOrder(root, ref sum, false);
            //return sum;
            int sum = 0;
            if (root != null)
            {
                if (root.left != null)
                {
                    if (root.left.right == null && root.left.left == null)
                        sum += root.left.val;
                    sum += SumOfLeftLeaves(root.left);
                }
                if (root.right != null)
                    sum += SumOfLeftLeaves(root.right);
            }
            return sum;
        }
        
        public static void PreOrder(TreeNode root, ref int sum, bool left)
        {
            if (root == null) return;
            if (left && root.left == null && root.right == null) sum += root.val;
            PreOrder(root?.left, ref sum, true);
            PreOrder(root?.right, ref sum, false);
        }
        #endregion

        #region LeetCode_371
        public int GetSum(int a, int b)
        {
            //a^b找不同的位置 即无需进位的位置
            int res = a ^ b;
            //a&b找同为1的位置 需要往左移一位
            int forward = (a & b) << 1;
            if (forward != 0)
                return GetSum(res, forward);
            return res;
        }
        #endregion


    }


}
