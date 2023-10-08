//
//  SpectrumAnalyzerPreferencesViewController.h
//  DeaDBeeF
//
//  Created by Oleksiy Yakovenko on 14/11/2021.
//  Copyright © 2021 Oleksiy Yakovenko. All rights reserved.
//

#import <Cocoa/Cocoa.h>

@class SpectrumAnalyzerSettings;

NS_ASSUME_NONNULL_BEGIN

@interface SpectrumAnalyzerPreferencesViewController : NSViewController

@property (nullable, nonatomic) SpectrumAnalyzerSettings *settings;
@property (nullable, nonatomic, weak) NSPopover *popover;

@end

NS_ASSUME_NONNULL_END
