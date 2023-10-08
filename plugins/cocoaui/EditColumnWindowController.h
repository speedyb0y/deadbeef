//
//  EditColumnWindowController.h
//  DeaDBeeF
//
//  Created by Oleksiy Yakovenko on 11/26/19.
//  Copyright © 2019 Oleksiy Yakovenko. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import "PlaylistViewController.h"

NS_ASSUME_NONNULL_BEGIN

@interface EditColumnWindowController : NSWindowController

@property (weak,nonatomic) IBOutlet NSTextField *titleTextField;
@property (weak,nonatomic) IBOutlet NSPopUpButton *typePopUpButton;
@property (weak,nonatomic) IBOutlet NSTextField *formatTextField;
@property (weak) IBOutlet NSTextField *sortFormatTextField;

@property (weak,nonatomic) IBOutlet NSPopUpButton *alignmentPopUpButton;
@property (weak,nonatomic) IBOutlet NSButton *setColorButton;
@property (weak,nonatomic) IBOutlet NSColorWell *colorWell;

- (void)initAddColumnSheet;

- (void)initEditColumnSheetWithTitle:(NSString *)title
                                type:(int)type
                              format:(NSString *)format
                              sortFormat:(NSString *)sortFormat
                           alignment:(PlaylistColumnAlignment)alignment
                        setTextColor:(BOOL)setTextColor
                           textColor:(NSColor *)textColor;

@end

NS_ASSUME_NONNULL_END
