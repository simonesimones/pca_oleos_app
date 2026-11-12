import io, zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, cut_tree
from core import preprocess,pca,diagnostics,correlations,hca
st.set_page_config(page_title='Toolbox | Quimiometria',layout='wide')
st.title('Toolbox de Quimiometria — PCA e HCA')
st.caption('Análise exploratória • variáveis suplementares não participam do ajuste.')

def read(f):
    if f.name.lower().endswith('.csv'):
        for enc in ['utf-8-sig','latin-1']:
            try: return pd.read_csv(io.BytesIO(f.getvalue()),sep=None,engine='python',encoding=enc)
            except (UnicodeError,pd.errors.ParserError): pass
        raise ValueError('CSV inválido.')
    return pd.read_excel(f)

def numeric(df,cols):
    return df[cols].apply(lambda s: pd.to_numeric(s.astype(str).str.strip().str.replace(',','.',regex=False),errors='coerce')).to_numpy(float)

def csv(df): return df.to_csv(index=True).encode('utf-8-sig')

f=st.sidebar.file_uploader('1. Dados principais',type=['xlsx','xls','csv'])
if not f:
    st.info('Carregue uma planilha com cabeçalhos e uma linha por amostra. Identificadores e classes são opcionais.')
    st.stop()
try: df=read(f)
except Exception as e: st.error(str(e)); st.stop()
cols=list(df.columns); st.dataframe(df.head(15))
idcol=st.sidebar.selectbox('Identificador da amostra',[None]+cols,format_func=lambda x:'Número da linha' if x is None else str(x))
classcol=st.sidebar.selectbox('Coluna de classes',[None]+[c for c in cols if c!=idcol],format_func=lambda x:'Sem classes' if x is None else str(x))
opts=[c for c in cols if c not in [idcol,classcol]]
active=st.sidebar.multiselect('Variáveis ativas',opts,default=[])
extra=st.sidebar.file_uploader('Arquivo suplementar (opcional)',type=['xlsx','xls','csv'])
try:
    if extra:
        ed=read(extra)
        if idcol is None: raise ValueError('Selecione um identificador para alinhar arquivos separados.')
        eid=st.sidebar.selectbox('Identificador no arquivo suplementar',list(ed.columns))
        if df[idcol].isna().any() or ed[eid].isna().any() or df[idcol].duplicated().any() or ed[eid].duplicated().any(): raise ValueError('Identificadores devem ser únicos e não vazios.')
        if set(df[idcol])!=set(ed[eid]): raise ValueError('Identificadores dos arquivos não coincidem.')
        ed=ed.set_index(eid).loc[df[idcol]].reset_index(drop=True)
        sup=st.sidebar.multiselect('Variáveis suplementares',list(ed.columns))
    else:
        ed=df; sup=st.sidebar.multiselect('Variáveis suplementares',[c for c in opts if c not in active])
except Exception as e: st.error(str(e)); st.stop()
if not active: st.info('Selecione as variáveis ativas na barra lateral.'); st.stop()
scale=st.sidebar.radio('2. Pré-processamento',['Autoescalamento','Centralização'])
weighted=st.sidebar.checkbox('Aplicar pesos individuais')
w=np.ones(len(active))
if weighted:
    wt=st.sidebar.data_editor(pd.DataFrame({'Variável':active,'Peso':w}),disabled=['Variável'],hide_index=True,key='weights_'+str(active))
    w=wt['Peso'].to_numpy(float)
try:
    z,mu,sd,w=preprocess(numeric(df,active),scale=='Autoescalamento',w)
    t,p,l,ev=pca(z)
    sx=numeric(ed,sup) if sup else None
    if sx is not None and not np.isfinite(sx).all(): raise ValueError('Variáveis suplementares contêm valores inválidos.')
except Exception as e: st.error(str(e)); st.stop()
n,r=t.shape; pcs=[f'PC{i+1}' for i in range(r)]
ids=df[idcol].astype(str).tolist() if idcol is not None else [str(i+1) for i in range(n)]
classes=df[classcol].fillna('(ausente)').astype(str).to_numpy() if classcol is not None else np.repeat('Amostras',n)
levels=list(dict.fromkeys(classes)); palette=plt.get_cmap('tab20'); markers=['o','s','^','D','v','P','X','*','<','>']
st.sidebar.header('3. Aparência')
font=st.sidebar.slider('Tamanho da fonte',8,22,11); size=st.sidebar.slider('Tamanho dos símbolos',10,200,45); labels=st.sidebar.checkbox('Exibir identificadores',True)
width=st.sidebar.slider('Largura (polegadas)',5,16,8); height=st.sidebar.slider('Altura (polegadas)',4,12,6); dpi=st.sidebar.selectbox('Resolução PNG',[300,600,1200])
styles={}
with st.sidebar.expander('Cores e símbolos por classe'):
    from matplotlib.colors import to_hex
    for i,c in enumerate(levels):
        styles[c]=(st.color_picker(c,to_hex(palette(i%20)),key=f'color{i}'),st.selectbox('Símbolo — '+c,markers,index=i%len(markers),key=f'marker{i}'))
plt.rcParams.update({'font.size':font,'svg.fonttype':'none','pdf.fonttype':42})
figs={}; tables={}
def figure(name):
    fig,ax=plt.subplots(figsize=(width,height)); figs[name]=fig; return fig,ax

def show(fig):
    fig.tight_layout(); st.pyplot(fig)

def scatter(ax,x,y):
    for c in levels:
        ix=classes==c; color,marker=styles[c]; ax.scatter(x[ix],y[ix],c=color,marker=marker,s=size,label=c,edgecolors='black',linewidths=.4)
    if labels:
        for i,s in enumerate(ids): ax.annotate(s,(x[i],y[i]),xytext=(4,4),textcoords='offset points',fontsize=max(7,font-2))
    if classcol is not None: ax.legend()

def arrows(ax,mat,names,color,ls='-'):
    for (x,y),name in zip(mat,names):
        if np.isfinite([x,y]).all():
            ax.annotate('',(x,y),(0,0),arrowprops={'arrowstyle':'->','color':color,'linestyle':ls,'lw':1.5}); ax.annotate(str(name),(x,y),xytext=(4,4),textcoords='offset points',color=color)

def axes_pc(ax,i,j):
    ax.axhline(0,color='0.8'); ax.axvline(0,color='0.8'); ax.set(xlabel=f'{pcs[i]} ({ev[i]:.1f}%)',ylabel=f'{pcs[j]} ({ev[j]:.1f}%)'); ax.set_aspect('equal',adjustable='datalim'); ax.margins(.2)

tables['scores']=pd.DataFrame(t,index=ids,columns=pcs); tables['loadings']=pd.DataFrame(p,index=active,columns=pcs); tables['variancia']=pd.DataFrame({'Autovalor':l,'Explicada (%)':ev,'Acumulada (%)':ev.cumsum()},index=pcs)
tables['preprocessamento']=pd.DataFrame({'Media':mu,'Escala':sd,'Peso':w},index=active); tables['X_processada']=pd.DataFrame(z,index=ids,columns=active)
ca=correlations(z,t); tables['correlacoes_ativas']=pd.DataFrame(ca,index=active,columns=pcs)
cs=correlations(sx,t) if sup else None
if sup: tables['correlacoes_suplementares']=pd.DataFrame(cs,index=sup,columns=pcs)
pa,diag,ht,export=st.tabs(['PCA','Diagnóstico','HCA','Exportação'])
with pa:
    st.write(f'{n} amostras · {len(active)} variáveis · {r} PCs de posto numérico não nulo')
    fig,ax=figure('variancia'); ax.bar(np.arange(1,r+1),ev,label='Individual'); ax.plot(np.arange(1,r+1),ev.cumsum(),'o-',label='Acumulada'); ax.set(xlabel='PC',ylabel='Variância (%)',ylim=(0,105)); ax.legend(); show(fig)
    st.dataframe(tables['variancia'])
    if r>=2:
        i=st.selectbox('PC horizontal',range(r),format_func=lambda k:pcs[k]); j=st.selectbox('PC vertical',[k for k in range(r) if k!=i],format_func=lambda k:pcs[k]); ij=[i,j]
        fig,ax=figure('scores'); scatter(ax,t[:,i],t[:,j]); axes_pc(ax,i,j); show(fig)
        fig,ax=figure('biplot_loadings'); scatter(ax,t[:,i],t[:,j]); factor=.75*np.max(np.abs(t[:,ij]))/max(np.max(np.abs(p[:,ij])),1e-15); arrows(ax,p[:,ij]*factor,active,'#145a32'); axes_pc(ax,i,j); ax.set_title('Biplot: loadings × fator gráfico'); show(fig)
        fig,ax=figure('circulo_correlacoes'); ax.add_patch(plt.Circle((0,0),1,fill=False,color='0.6')); arrows(ax,ca[:,ij],active,'#145a32');
        if sup: arrows(ax,cs[:,ij],sup,'#a93226','--')
        axes_pc(ax,i,j); ax.set(xlim=(-1.25,1.25),ylim=(-1.25,1.25),title='Correlações: ativas (verde), suplementares (vermelho)'); show(fig)
        fig,ax=figure('biplot_suplementar'); ts=t[:,ij]/np.sqrt(l[ij]); scatter(ax,ts[:,0],ts[:,1]); arrows(ax,ca[:,ij],active,'#145a32');
        if sup: arrows(ax,cs[:,ij],sup,'#a93226','--')
        axes_pc(ax,i,j); ax.set_title('Scores normalizados e vetores de correlação'); show(fig)
    else: st.info('Apenas uma PC disponível; gráficos bidimensionais não se aplicam.')
    k=st.slider('PCs no mapa de loadings',1,r,min(r,5)) if r>1 else 1
    fig,ax=figure('loadings'); im=ax.imshow(p[:,:k].T,aspect='auto',cmap='RdBu_r',vmin=-1,vmax=1); ax.set_xticks(range(len(active)),labels=active,rotation=45,ha='right'); ax.set_yticks(range(k),labels=pcs[:k]); fig.colorbar(im,ax=ax); show(fig)
    ordercol=st.selectbox('Ordenar evolução dos scores por',[None]+cols,format_func=lambda x:'Ordem original' if x is None else str(x)); order=np.arange(n) if ordercol is None else df[ordercol].sort_values(kind='stable').index.to_numpy(); pc=st.selectbox('PC para evolução',range(r),format_func=lambda k:pcs[k]); fig,ax=figure('evolucao'); ax.plot(np.arange(n),t[order,pc],'o-'); ax.set_xticks(range(n),labels=np.asarray(ids)[order],rotation=45,ha='right'); ax.set_ylabel(pcs[pc]); show(fig)
with diag:
    a=st.slider('PCs retidas no diagnóstico',1,r,min(2,r)) if r>1 else 1
    conf=st.selectbox('Quantil empírico',[.90,.95,.99],index=1)
    T,Q,lt,lq=diagnostics(z,t,p,l,a,conf)
    st.warning('Limites empíricos calculados nas próprias amostras: sinalização exploratória, não teste formal nem limite validado para novas amostras. Não exclua amostras automaticamente.')
    if a==r: st.info('Todas as PCs retidas: Q é numericamente zero e não discrimina resíduos.')
    flag=(T>lt)|((Q>lq) if a<r else False)
    tables['diagnostico']=pd.DataFrame({'T2':T,'Q':Q,'Limite T2':lt,'Limite Q':lq,'Sinalizada':flag},index=ids); st.dataframe(tables['diagnostico'])
    fig,ax=figure('T2_Q'); scatter(ax,T,Q); ax.axvline(lt,color='red',ls='--');
    if a<r: ax.axhline(lq,color='red',ls='--')
    ax.set(xlabel='Hotelling T²',ylabel='Resíduo Q'); show(fig)
    for name,v,lim in [('T2',T,lt),('Q',Q,lq)]:
        fig,ax=figure(name); scatter(ax,np.arange(n),v); ax.axhline(lim,color='red',ls='--'); ax.set(xlabel='Índice da amostra (base 0)',ylabel=name); show(fig)
with ht:
    st.caption('HCA sobre a matriz pré-processada completa, não sobre os scores truncados.')
    method=st.selectbox('Ligação',['ward','average','complete','single','weighted']); metric=st.selectbox('Distância',['euclidean'] if method=='ward' else ['euclidean','cityblock','cosine','correlation','chebyshev']); groups=st.number_input('Número de grupos',1,n,min(3,n))
    try:
        link=hca(z,method,metric); membership=cut_tree(link,n_clusters=[groups]).ravel()+1
        fig,ax=figure('dendrograma'); dendrogram(link,labels=ids,ax=ax,leaf_rotation=90,color_threshold=0,above_threshold_color='#333333'); ax.set_ylabel('Distância de ligação'); show(fig)
        tables['HCA_grupos']=pd.DataFrame({'Grupo':membership,'Classe informada':classes},index=ids); tables['HCA_ligacoes']=pd.DataFrame(link,columns=['No1','No2','Distancia','N']); st.dataframe(tables['HCA_grupos'])
    except Exception as e: st.error('HCA: '+str(e))
with export:
    st.caption('PNG na resolução selecionada; SVG e PDF vetoriais. Tabelas CSV e parâmetros de processamento.')
    fmt=st.multiselect('Formatos de figuras',['png','svg','pdf'],default=['png','svg','pdf'])
    tables['configuracao']=pd.DataFrame({'Valor':[scale,str(weighted),a,conf,method,metric,groups]},index=['Preprocessamento','Ponderacao','PCs diagnostico','Quantil','Ligacao HCA','Distancia HCA','Grupos HCA'])
    if st.button('Preparar arquivo ZIP'):
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as out:
            for name,tab in tables.items(): out.writestr(name+'.csv',csv(tab))
            for name,fig in figs.items():
                for ext in fmt:
                    b=io.BytesIO(); fig.savefig(b,format=ext,dpi=dpi,bbox_inches='tight'); out.writestr(name+'.'+ext,b.getvalue())
        st.session_state['zip_result']=buf.getvalue()
    if 'zip_result' in st.session_state: st.download_button('Baixar ZIP preparado',st.session_state['zip_result'],'resultados_quimiometria.zip',mime='application/zip'); st.caption('Após mudar configurações, prepare o ZIP novamente.')
plt.close('all')
