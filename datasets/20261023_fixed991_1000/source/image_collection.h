#pragma once
// Same STB decode/resize, patch size, scaling and xy ordering as LiteRT-LM's
// stb_image_preprocessor.cc and image_preprocessor_utils.cc (Gemma4 defaults).
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
struct CollectionImage { Bytes patches,positions;int rows,width,height; };
static CollectionImage collection_image(const Bytes& encoded){
 require(!encoded.empty()&&encoded.size()<32*1024*1024,"Invalid encoded image size");
 int w,h,ch;auto* raw=stbi_load_from_memory(encoded.data(),int(encoded.size()),&w,&h,&ch,3);
 require(raw!=nullptr,"Cannot decode training image");
 std::unique_ptr<unsigned char,void(*)(void*)> guard(raw,stbi_image_free);
 require(w>0&&h>0&&int64_t(w)*h<=100000000,"Training image dimensions exceed limit");
 float factor=std::sqrt(float(2520*256)/(float(w)*h));
 int nh=int(std::floor(factor*h/48))*48,nw=int(std::floor(factor*w/48))*48;
 if(nh==0){nh=48;nw=std::min(int(std::floor(float(w)/h))*48,280*48);}
 else if(nw==0){nw=48;nh=std::min(int(std::floor(float(h)/w))*48,280*48);}
 require(nh>0&&nw>0&&nh*nw<=2520*256,"Invalid resized image");
 std::vector<unsigned char> resized(size_t(nw)*nh*3);
 if(nw==w&&nh==h)std::copy(raw,raw+resized.size(),resized.begin());
 else require(stbir_resize(raw,w,h,0,resized.data(),nw,nh,0,STBIR_RGB,STBIR_TYPE_UINT8_SRGB,STBIR_EDGE_CLAMP,STBIR_FILTER_CATMULLROM)!=0,"Image resize failed");
 std::vector<float> patches(2520*768,0);std::vector<int32_t> positions(2520*2,-1);
 int pw=nw/16,ph=nh/16;
 for(int y=0;y<ph;y++)for(int x=0;x<pw;x++){
  size_t p=size_t(y)*pw+x;positions[2*p]=x;positions[2*p+1]=y;
  for(int a=0;a<16;a++)for(int b=0;b<16;b++)for(int c=0;c<3;c++)
   patches[p*768+(a*16+b)*3+c]=float(resized[((y*16+a)*nw+x*16+b)*3+c])*(1.f/255.f);
 }
 return {bytes(patches),bytes(positions),pw*ph/9,nw,nh};
}
