//
//  SegmentedTabView.h
//  deadbeef
//
//  Created by Oleksiy Yakovenko on 19/11/2021.
//  Copyright © 2021 Oleksiy Yakovenko. All rights reserved.
//

#import <Cocoa/Cocoa.h>

NS_ASSUME_NONNULL_BEGIN

/// Wrap NSTabView and NSSegmentedControl into a single view, and use the segmented control for displaying tabs.
@interface SegmentedTabView : NSControl

@property (readonly) NSInteger numberOfTabViewItems;
@property (nonatomic) NSArray<__kindof NSTabViewItem *> *tabViewItems;
@property (nullable, readonly) NSTabViewItem *selectedTabViewItem;

- (void)removeTabViewItem:(NSTabViewItem *)tabViewItem;
- (void)addTabViewItem:(NSTabViewItem *)tabViewItem;
- (void)insertTabViewItem:(NSTabViewItem *)tabViewItem atIndex:(NSInteger)index;
- (nullable NSTabViewItem *)tabViewItemAtPoint:(NSPoint)point;
- (NSInteger)indexOfTabViewItem:(NSTabViewItem *)tabViewItem;
- (NSTabViewItem *)tabViewItemAtIndex:(NSInteger)index;
- (void)setLabel:(NSString *)label forSegment:(NSInteger)index;
- (void)selectTabViewItemAtIndex:(NSInteger)index;

@end

NS_ASSUME_NONNULL_END
