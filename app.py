import io
import zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='PCA | Óleos essenciais', layout='wide')
st.title('PCA e biplots — caracterização temporal de óleos essenciais')
st.caption('Compostos como variáveis ativas; ambiente como variáveis suplementares (correlações com os scores).')
MESES = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
COLORS = ['#337fd9','#2666cc','#40a659','#4db34d','#80bf33','#d9a61a','#e6661a','#d94026','#cc4d33','#b38026','#6699bf','#3373cc']

def load_file(f):
    if f.name.lower().endswith('.csv'):
        raw=f.getvalue()
        for enc in ['utf-8-sig','latin-1']:
            try:
                return pd.read_csv(io.BytesIO(raw), sep=None, engine='python', encoding=enc)
            except (UnicodeError, pd.errors.ParserError, ValueError):
                continue
        raise ValueError('CSV não reconhecido. Confira delimitador e codificação.')
    return pd.read_excel(f)

def numcols(df, columns):
    out=df[columns].copy()
    for col in columns:
        if out[col].dtype == 'object':
            out[col]=out[col].astype(str).str.strip().str.replace(',','.',regex=False)
        out[col]=pd.to_numeric(out[col], errors='coerce')
    return out

def make_pca(x, scaling):
    mu=x.mean(axis=0)
    sd=x.std(axis=0,ddof=1)
    if np.any(sd<=1e-12):
        raise ValueError('Há compostos constantes. Remova ou revise essas colunas.')
    z=(x-mu)/sd if scaling=='Autoescalamento (z-score)' else x-mu
    u,s,vt=np.linalg.svd(z,full_matrices=False)
    coeff=vt.T
    scores=u*s
    latent=s**2/(len(x)-1)
    explained=100*latent/latent.sum()
    return z,coeff,scores,latent,explained

def points(fig,scores,months):
    for i,m in enumerate(months):
        fig.add_trace(go.Scatter(x=[scores[i,0]],y=[scores[i,1]],mode='markers+text',text=[MESES[m-1]],textposition='top center',marker=dict(size=12,color=COLORS[m-1],line=dict(color='black',width=0.7)),name=MESES[m-1],showlegend=False))

def arrow(fig,x,y,name,color,dash='solid'):
    fig.add_annotation(x=x,y=y,ax=0,ay=0,xref='x',yref='y',axref='x',ayref='y',showarrow=True,arrowhead=3,arrowsize=1,arrowwidth=2,arrowcolor=color,text='')
    fig.add_annotation(x=x*1.12,y=y*1.12,text=name,showarrow=False,font=dict(color=color,size=12))
    if dash=='dash':
        fig.add_trace(go.Scatter(x=[0,x],y=[0,y],mode='lines',line=dict(color=color,width=2,dash='dash'),showlegend=False,hoverinfo='skip'))

def style_xy(fig, explained, title):
    fig.update_layout(title=title,xaxis_title=f'PC1 ({explained[0]:.1f}%)',yaxis_title=f'PC2 ({explained[1]:.1f}%)',template='plotly_white',height=590)
    fig.add_hline(y=0,line_color='lightgray');fig.add_vline(x=0,line_color='lightgray')
    return fig

st.sidebar.header('1. Importação')
mode=st.sidebar.radio('Formato dos dados',['Arquivo único','Arquivos separados'])
if mode=='Arquivo único':
    f=st.sidebar.file_uploader('CSV ou Excel: meses + compostos + ambiente',type=['csv','xlsx','xls'])
    if not f:
        st.info('Carregue um arquivo para iniciar. Uma linha por amostra; inclua uma coluna de mês, colunas de compostos e, opcionalmente, colunas ambientais.')
        st.dataframe(pd.DataFrame({'Mes':list(range(1,13)),'Biciclogermacreno':['']*12,'D-Germacreno':['']*12,'E-Cariofileno':['']*12,'Temperatura':['']*12,'Umidade':['']*12,'Pluviosidade':['']*12,'Luminosidade':['']*12}))
        st.stop()
    try: df=load_file(f)
    except Exception as e: st.error(str(e));st.stop()
    env_df=df
else:
    fc=st.sidebar.file_uploader('Arquivo dos compostos',type=['csv','xlsx','xls'],key='comp')
    fe=st.sidebar.file_uploader('Arquivo ambiental (opcional)',type=['csv','xlsx','xls'],key='amb')
    if not fc: st.info('Carregue o arquivo dos compostos para iniciar.');st.stop()
    try:
        df=load_file(fc)
        env_df=load_file(fe) if fe else None
    except Exception as e: st.error(str(e));st.stop()

st.subheader('Prévia dos dados de compostos')
st.dataframe(df.head(15),use_container_width=True)
cols=list(df.columns)
st.sidebar.header('2. Seleção das colunas')
month_col=st.sidebar.selectbox('Coluna de mês',cols,index=next((i for i,c in enumerate(cols) if str(c).lower() in ['mes','mês','month']),0))
comp_defaults=[c for c in cols if c!=month_col][:3]
comp_cols=st.sidebar.multiselect('Compostos ativos', [c for c in cols if c!=month_col],default=comp_defaults)
if env_df is not None:
    envcols=list(env_df.columns)
    env_month_col=st.sidebar.selectbox('Coluna de mês no arquivo ambiental',envcols,index=next((i for i,c in enumerate(envcols) if str(c).lower() in ['mes','mês','month']),0)) if mode=='Arquivos separados' else month_col
    amb_options=[c for c in envcols if c!=env_month_col and c not in (comp_cols if mode=='Arquivo único' else [])]
    env_defaults=[c for c in amb_options if any(k in str(c).lower() for k in ['temp','umid','pluv','lumi'])]
    env_cols=st.sidebar.multiselect('Variáveis ambientais suplementares',amb_options,default=env_defaults)
else:
    env_cols=[]
scaling=st.sidebar.radio('Pré-processamento dos compostos',['Autoescalamento (z-score)','Centralização na média'])
if len(comp_cols)<2: st.warning('Selecione pelo menos dois compostos.');st.stop()

try:
    months=pd.to_numeric(df[month_col],errors='raise').to_numpy()
    if not np.all(np.isfinite(months)) or not np.all(np.isin(months,np.arange(1,13))):
        raise ValueError('Os meses devem ser inteiros de 1 a 12, sem valores ausentes.')
    months=months.astype(int)
    if len(np.unique(months))!=len(months):
        raise ValueError('Há meses repetidos. Esta versão exige uma amostra por mês.')
    x=numcols(df,comp_cols).to_numpy(dtype=float)
    if not np.isfinite(x).all(): raise ValueError('Há valores ausentes ou não numéricos nos compostos.')
    if len(x)<3: raise ValueError('São necessárias pelo menos três amostras.')
    env=None
    if env_cols:
        if mode=='Arquivos separados':
            envmonths=pd.to_numeric(env_df[env_month_col],errors='raise').to_numpy()
            if not np.all(np.isfinite(envmonths)) or not np.all(np.isin(envmonths,np.arange(1,13))) or len(set(envmonths))!=len(envmonths):
                raise ValueError('Meses ambientais inválidos ou duplicados.')
            ed=env_df.copy();ed['_mes_chave_']=envmonths.astype(int)
            if not set(months).issubset(set(ed['_mes_chave_'])):
                raise ValueError('O arquivo ambiental não contém todos os meses dos compostos.')
            env=numcols(ed.set_index('_mes_chave_').loc[months].reset_index(),env_cols).to_numpy(dtype=float)
        else:
            env=numcols(env_df,env_cols).to_numpy(dtype=float)
        if not np.isfinite(env).all(): raise ValueError('Há valores ausentes ou não numéricos nas variáveis ambientais.')
    z,coeff,scores,latent,explained=make_pca(x,scaling)
    corrs=None
    if env is not None:
        if np.any(env.std(axis=0,ddof=1)<=1e-12): raise ValueError('Há variáveis ambientais constantes.')
        corrs=np.array([[np.corrcoef(env[:,j],scores[:,k])[0,1] for k in range(2)] for j in range(len(env_cols))])
except Exception as e:
    st.error(f'Verifique os dados: {e}');st.stop()

st.success(f'{len(x)} amostras | {len(comp_cols)} compostos | PC1 + PC2 = {explained[:2].sum():.2f}% da variância')
st.caption('A PCA é exploratória. Associações ambientais são correlações descritivas, não evidências de causalidade.')
figures={}
scale=0.75*np.max(np.abs(scores[:,:2]))/max(np.max(np.abs(coeff[:,:2])),1e-12)
fig1=go.Figure();points(fig1,scores,months)
for j,c in enumerate(comp_cols): arrow(fig1,coeff[j,0]*scale,coeff[j,1]*scale,str(c),'#166534')
style_xy(fig1,explained,'Biplot — compostos ativos (setas com escala gráfica)')
figures['biplot_compostos']=fig1
fig2=go.Figure();points(fig2,scores,months)
for j,c in enumerate(comp_cols):arrow(fig2,coeff[j,0]*scale,coeff[j,1]*scale,str(c),'#166534')
if corrs is not None:
    for j,c in enumerate(env_cols):arrow(fig2,corrs[j,0]*scale*.85,corrs[j,1]*scale*.85,str(c),'#b45309','dash')
style_xy(fig2,explained,'Biplot com ambiente suplementar (setas laranja tracejadas)')
figures['biplot_ambiente']=fig2
fig3=go.Figure();fig3.add_bar(x=[f'PC{i+1}' for i in range(len(explained))],y=explained,name='Individual')
fig3.add_scatter(x=[f'PC{i+1}' for i in range(len(explained))],y=np.cumsum(explained),mode='lines+markers',name='Acumulada',yaxis='y2')
fig3.update_layout(title='Variância explicada',yaxis=dict(title='Individual (%)'),yaxis2=dict(title='Acumulada (%)',overlaying='y',side='right',range=[0,105]),template='plotly_white',height=470)
figures['variancia_explicada']=fig3
fig4=go.Figure(data=go.Heatmap(z=coeff.T,x=[str(c) for c in comp_cols],y=[f'PC{i+1}' for i in range(coeff.shape[1])],colorscale='RdBu',reversescale=True,zmid=0,zmin=-1,zmax=1,text=np.round(coeff.T,3),texttemplate='%{text}',colorbar=dict(title='Coef.')))
fig4.update_layout(title='Coeficientes das componentes (loadings)',height=420)
figures['heatmap_loadings']=fig4
order=np.argsort(months)
fig5=go.Figure()
for k in range(2):fig5.add_scatter(x=[MESES[m-1] for m in months[order]],y=scores[order,k],mode='lines+markers',name=f'PC{k+1} ({explained[k]:.1f}%)')
fig5.update_layout(title='Scores por mês (ordem cronológica)',xaxis_title='Mês',yaxis_title='Score',template='plotly_white',height=470)
figures['scores_mensais']=fig5

tabs=st.tabs(['Biplot compostos','Biplot + ambiente','Scree plot','Loadings','Scores mensais','Tabelas e exportação'])
for tab,key in zip(tabs[:5],list(figures)):
    with tab:st.plotly_chart(figures[key],use_container_width=True)
with tabs[5]:
    st.markdown('**Variância explicada**');st.dataframe(pd.DataFrame({'PC':[f'PC{i+1}' for i in range(len(explained))],'Autovalor':latent,'Variância (%)':explained,'Acumulada (%)':np.cumsum(explained)}),use_container_width=True)
    st.markdown('**Coeficientes das componentes**');st.dataframe(pd.DataFrame(coeff,index=comp_cols,columns=[f'PC{i+1}' for i in range(coeff.shape[1])]),use_container_width=True)
    st.markdown('**Scores**');st.dataframe(pd.DataFrame(scores,index=[MESES[m-1] for m in months],columns=[f'PC{i+1}' for i in range(scores.shape[1])]),use_container_width=True)
    if corrs is not None:
        st.markdown('**Correlações ambientais com scores**');st.dataframe(pd.DataFrame(corrs,index=env_cols,columns=['PC1','PC2']),use_container_width=True)
    outputs={'variancia.csv':pd.DataFrame({'PC':np.arange(1,len(explained)+1),'autovalor':latent,'variancia_percentual':explained}).to_csv(index=False),'loadings.csv':pd.DataFrame(coeff,index=comp_cols).to_csv(),'scores.csv':pd.DataFrame(scores,index=months).to_csv()}
    if corrs is not None:outputs['correlacoes_ambiente.csv']=pd.DataFrame(corrs,index=env_cols,columns=['PC1','PC2']).to_csv()
    for name,fig in figures.items():outputs[name+'.html']=fig.to_html(full_html=True,include_plotlyjs=True)
    buff=io.BytesIO()
    with zipfile.ZipFile(buff,'w',zipfile.ZIP_DEFLATED) as zf:
        for name,content in outputs.items():zf.writestr(name,content)
    st.download_button('Baixar resultados e gráficos interativos (.zip)',buff.getvalue(),file_name='resultados_PCA_oleos.zip',mime='application/zip')
st.caption('Nota metodológica: as setas de compostos representam autovetores reescalados para exibição; as ambientais representam correlações reescaladas. Seus comprimentos não devem ser comparados diretamente.')
