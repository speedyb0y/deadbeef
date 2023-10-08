#import <Cocoa/Cocoa.h>
#import "SpectrumAnalyzerSettings.h"
#import "VisualizationViewController.h"

@class SpectrumAnalyzerSettings;

@interface SpectrumAnalyzerVisualizationViewController : VisualizationViewController

@property (nonatomic,nullable) SpectrumAnalyzerSettings *settings;

- (void)message:(uint32_t)_id ctx:(uintptr_t)ctx p1:(uint32_t)p1 p2:(uint32_t)p2;
- (void)updateAnalyzerSettings:(SpectrumAnalyzerSettings * _Nonnull)settings;

@end
