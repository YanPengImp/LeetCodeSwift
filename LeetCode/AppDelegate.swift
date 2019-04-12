//
//  AppDelegate.swift
//  LeetCode
//
//  Created by Imp on 2018/6/28.
//  Copyright © 2018年 jingbo. All rights reserved.
//

import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?


    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplicationLaunchOptionsKey: Any]?) -> Bool {
        let l = LeetCode()
        let result = l.findDuplicates([4,3,2,7,8,2,3,1])
        print(result)

        
        //二叉树
//        let root = LeetCode.TreeNode.init(1)
//        root.left = LeetCode.TreeNode.init(2)
//        root.right = LeetCode.TreeNode.init(3)
//        root.left?.left = LeetCode.TreeNode.init(4)
//        root.right?.right = LeetCode.TreeNode.init(5)
//        l.maxDepth(root)

        //链表
//        let head = LeetCode.ListNode.init(1)
//        head.next = LeetCode.ListNode.init(2)
//        head.next?.next = LeetCode.ListNode.init(3)
//        head.next?.next?.next = LeetCode.ListNode.init(4)
//        l.removeElements(head, 2)

        // Override point for customization after application launch.
        return true
    }

    func applicationWillResignActive(_ application: UIApplication) {
        // Sent when the application is about to move from active to inactive state. This can occur for certain types of temporary interruptions (such as an incoming phone call or SMS message) or when the user quits the application and it begins the transition to the background state.
        // Use this method to pause ongoing tasks, disable timers, and invalidate graphics rendering callbacks. Games should use this method to pause the game.
    }

    func applicationDidEnterBackground(_ application: UIApplication) {
        // Use this method to release shared resources, save user data, invalidate timers, and store enough application state information to restore your application to its current state in case it is terminated later.
        // If your application supports background execution, this method is called instead of applicationWillTerminate: when the user quits.
    }

    func applicationWillEnterForeground(_ application: UIApplication) {
        // Called as part of the transition from the background to the active state; here you can undo many of the changes made on entering the background.
    }

    func applicationDidBecomeActive(_ application: UIApplication) {
        // Restart any tasks that were paused (or not yet started) while the application was inactive. If the application was previously in the background, optionally refresh the user interface.
    }

    func applicationWillTerminate(_ application: UIApplication) {
        // Called when the application is about to terminate. Save data if appropriate. See also applicationDidEnterBackground:.
    }


}

