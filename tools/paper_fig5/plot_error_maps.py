#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One 3x7 error-map grid, with one reference and one dual-calibrated colorbar."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from error_map_inputs import validate_bundle, font_setup, file_hash, PLANES

SOURCE_HASH='004226a35db39de2006f54ac563fd046ea5923b3cce85d7bfdc23c67539dff29'
NATIVE_METHODS=('TV_z','Chung','Lahiri','TV-LoRA')
METHODS=('ADMM-TV','FBPConvNet')+NATIVE_METHODS
ERROR_X=((899,1343),(2505,3078),(4176,4748))
LEGACY_Y={'ADMM-TV':(513,957),'FBPConvNet':(973,1416)}
ORIGINAL_BAR=(4970,204,4990,2086)
ORIGINAL_TICKS=((0.00,2085),(0.05,1796),(0.10,1508),(0.15,1220),
                (0.20,931),(0.25,643),(0.30,355))

def image_axis(fig,x,y,w,h):
    fw,fh=fig.get_size_inches()
    ax=fig.add_axes((x/fw,y/fh,w/fw,h/fh))
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.22);spine.set_color('#d8dde2')
    return ax

def render(base,output,dpi=600):
    source,arrays=validate_bundle(base/'error_sections',NATIVE_METHODS)
    original_path=base/'source_figures/ldct_error_maps_view8_original.png'
    if file_hash(original_path)!=SOURCE_HASH:
        raise ValueError('Original image changed; pinned rectangles not applied')
    original=Image.open(original_path).convert('RGB')
    if original.size!=(5171,2338):
        raise ValueError('Unexpected original image dimensions')
    font_setup()
    width=7.16
    left,right,ref_gap,gap=0.23,0.12,0.06,0.025
    cell=(width-left-right-ref_gap-5*gap)/7
    top,bottom,row_gap=0.28,0.54,0.035
    row_heights=[cell*r for r in (1,198/256,198/256)]
    height=top+bottom+sum(row_heights)+2*row_gap
    fig=plt.figure(figsize=(width,height),facecolor='white')
    xs=[left,left+cell+ref_gap]
    for unused in range(5):xs.append(xs[-1]+cell+gap)
    records,stats=[],[]
    y=height-top
    for r,plane in enumerate(PLANES):
        y-=row_heights[r]
        fig.text(0.10/width,(y+row_heights[r]/2)/height,plane.capitalize(),
                 rotation=90,ha='center',va='center',fontsize=8,color='#252c33')
        ref=arrays['reference__'+plane]
        image_axis(fig,xs[0],y,cell,row_heights[r]).imshow(
            np.flipud(ref),cmap='gray',vmin=0,vmax=1,
            aspect='equal',interpolation='nearest')
        for c,method in enumerate(METHODS,1):
            if method in LEGACY_Y:
                x0,x1=ERROR_X[r];y0,y1=LEGACY_Y[method]
                rect=(x0,y0,x1,y1)
                image_axis(fig,xs[c],y,cell,row_heights[r]).imshow(
                    np.asarray(original.crop(rect)),aspect='auto',interpolation='nearest')
                records.append(dict(method=method,plane=plane,rectangle=list(rect),
                                    recolored=False,scalar_errors_recovered=False))
            else:
                error=np.abs(arrays[method+'__'+plane].astype(np.float64)-ref.astype(np.float64))
                image_axis(fig,xs[c],y,cell,row_heights[r]).imshow(
                    np.flipud(error),cmap='magma',vmin=0,vmax=0.35,
                    aspect='equal',interpolation='nearest')
                stats.append(dict(method=method,plane=plane,
                                  section_mae=float(error.mean()),
                                  fraction_above_display_max=float(np.mean(error>0.35))))
        y-=row_gap
    labels=('Reference','ADMM-TV','FBPConvNet',r'$\mathrm{TV}_z$',
            'Chung','Lahiri','TV-LoRA')
    for x,name in zip(xs,labels):
        fig.text((x+cell/2)/width,(height-top+0.075)/height,name,
                 ha='center',va='bottom',fontsize=8.2,color='#252c33',
                 fontweight='bold' if name=='TV-LoRA' else 'normal')
    # One centered gradient; its two numeric calibrations are identified in the caption.
    # Rotate the original gradient only; do not alter panel colors or infer its upper limit.
    grid_left,grid_right=xs[0],xs[-1]+cell
    bar_width=0.82*(grid_right-grid_left)
    bar_left=(grid_left+grid_right-bar_width)/2
    bar_right=bar_left+bar_width
    cax=fig.add_axes((bar_left/width,0.30/height,
                     bar_width/width,0.045/height))
    strip=np.rot90(np.asarray(original.crop(ORIGINAL_BAR)),k=-1)
    cax.imshow(strip,extent=(0,1,0,1),origin='upper',aspect='auto',interpolation='nearest')
    cax.set_yticks([])
    native_ticks=(0.00,0.10,0.20,0.30,0.35)
    cax.set_xticks([v/0.35 for v in native_ticks],['%.2f'%v for v in native_ticks])
    cax.tick_params(axis='x',labelsize=7.2,length=1.7,width=0.35,pad=2.0)
    for spine in cax.spines.values():
        spine.set_linewidth(0.30);spine.set_color('#59636d')
    upper=cax.twiny()
    upper.set_xlim(0,1)
    original_display_ticks=[item for item in ORIGINAL_TICKS if item[0] in (0.00,0.10,0.20,0.30)]
    positions=[(ORIGINAL_BAR[3]-py)/(ORIGINAL_BAR[3]-ORIGINAL_BAR[1])
               for value,py in original_display_ticks]
    upper.set_xticks(positions,['%.2f'%value for value,py in original_display_ticks])
    upper.tick_params(axis='x',labelsize=7.2,length=1.7,width=0.35,pad=2.0)
    upper.patch.set_visible(False)
    for spine in upper.spines.values():spine.set_visible(False)
    centre=(bar_left+bar_right)/2/width
    fig.text(centre,0.025/height,'Absolute error',
             ha='center',va='bottom',fontsize=7.4,color='#252c33')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for item in fig.findobj(match=plt.Text):
        if item.get_visible() and item.get_text():
            b=item.get_window_extent(renderer)
            if b.x0<1 or b.y0<1 or b.x1>fig.bbox.width-1 or b.y1>fig.bbox.height-1:
                raise ValueError('Text exceeds figure boundary: '+item.get_text())
    output.mkdir(parents=True,exist_ok=True)
    paths=[]
    for extension in ('png','pdf','svg'):
        path=output/('Fig5_LDCT_8views.'+extension)
        fig.savefig(path,dpi=dpi,facecolor='white',pad_inches=0,
                    metadata={'Creator':'Matplotlib'} if extension=='pdf' else None)
        paths.append(path)
    plt.close(fig)
    report=dict(figure='Fig. 5',dataset='LDCT',case='L067',views=8,
        methods=list(METHODS),layout='one 3x7 grid; no subfigures',
        reference_columns=1,reference_column_applies_to=list(NATIVE_METHODS),
        physical_colorbar_count=1,numeric_colorbar_calibrations=2,
        upper_ticks_apply_to=['ADMM-TV','FBPConvNet'],
        lower_ticks_apply_to=list(NATIVE_METHODS),
        repeated_method_lists_below_panels=False,
        scale_assignment_stated_in_caption=True,
        displayed_upper_tick_values=[value for value,py in original_display_ticks],
        displayed_lower_tick_values=list(native_ticks),
        font_family='DejaVu Sans',size_inches=[width,height],png_dpi=dpi,
        original_source_sha256=SOURCE_HASH,original_panel_records=records,
        original_colorbar_rectangle=list(ORIGINAL_BAR),
        original_tick_values_and_source_y_pixels=list(ORIGINAL_TICKS),
        legacy_scalar_errors_reconstructed_from_RGB=False,
        legacy_gradient_and_panel_colors_unchanged=True,
        original_colorbar_upper_limit_inferred=False,
        native_section_archive_sha256=source['section_archive_sha256'],
        native_reference_window=[0,1],native_error_window=[0,0.35],
        native_methods_use_unmodified_saved_evaluation_outputs=True,
        all_methods_claimed_to_share_numeric_scale_or_reference=False,
        native_section_statistics=stats,outputs={p.name:file_hash(p) for p in paths})
    (output/'fig5_render_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for path in paths:print('FIGURE='+str(path))
    return report

def main():
    base=Path(__file__).resolve().parent.parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=base)
    parser.add_argument('--output',type=Path,default=base/'figures')
    parser.add_argument('--dpi',type=int,default=600)
    args=parser.parse_args()
    render(args.base,args.output,args.dpi)

if __name__=='__main__':
    main()

