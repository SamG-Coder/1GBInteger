"""Independent exact-rational assembly check; no upstream imports.

This checks the parameter refinement, not inherited geometric/all-size proofs.
"""
import copy,json,pathlib,sys
from fractions import Fraction as F
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
HERE=pathlib.Path(__file__).resolve().parent
def validate(d):
    r=d['best']['record'];A=d['best']['assembly'];B=d['best']['finite_bridge']
    coarse=F(d['refined_coarse_control']['coarse']);old=F(B['bit_uniform']['old_atom_saving'])
    atom=F(r['atom']);eta=F(r['eta']);k=F(r['kappa']);a=(1-atom)*coarse+atom*old
    assert 0<a<atom<1-a,'subordinate adapter/row constraint'
    assert a==F(r['stopped']) and atom-a==F(r['adapter_gap'])
    par=A['parameters'];b=F(par['a_complex']);beta=F(par['beta'])
    tau=1-a;sigma=1-b;q=a*(1-2*eta);lp=1-q;lam=(tau+lp)/2
    c=q*(1+eta);eps=(1-eta)/(1+c+q);minimum=eps*q
    alpha=(minimum+1-eps)/2;delta=eta/8
    internal=tau+(1-beta)*max(sigma-tau,F(0));leaf=sigma+beta*(1-sigma)
    margins={'original_prefix':1-eps*(1+c),'coordinate_movement':a,'compact_phase_layer':minimum,
        'bulk_exposure':a,'Gaussian_arithmetic':min(1-eps-delta,alpha-delta),
        'scalar_work':1-eps-delta,'dimension':eps}
    slacks=dict(bit_positive=a,complex_above_bit=b-a,complex_below_one_over32=F(1,32)-b,
        beta_positive=beta,beta_below_one=1-beta,leaf_saving_above_bit=(1-beta)*b-a,
        q_positive=q,q_below_internal=1-internal-q,q_below_leaf=1-leaf-q,
        c_positive=c,c_below_one=1-c,q_below_reservations=c-q,
        lambda_above_tau=lam-tau,lambda_above_sigma=lam-sigma,lambda_above_internal=lam-internal,
        lambda_prime_above_lambda=lp-lam,compact_leaf=lp-leaf,compact_reservations=lp-(1-c),
        lambda_prime_below_one=q,epsilon_positive=eps,epsilon_below_one=1-eps,
        guard_width=1-eps,K_geometry=1-eps*(1+c),K_dominates_log=eps*c,
        record_suffix=1-eps,phase_local=1-eps-delta,phase_boundary=alpha-delta,
        gamma_sublinear=1-eps-alpha,cell_above_band=eps-(1-alpha)/2,
        prime_interval_packing=1-eps,alpha_positive=alpha,alpha_below_one=1-alpha,
        alpha_below_one_fourth=F(1,4)-alpha,delta_positive=delta,delta_below_one_eighth=F(1,8)-delta,
        short_record_fallback=eps-a,small_field_exposure=1-eps-minimum,
        artificial_boundary=8-eps+alpha-delta-minimum,
        literal_scalar_guard=F(B['semantic']['strict_literal_gap']),row_product_gap=F(B['rows']['degree_gap']))
    slacks.update({name+'_above_kappa':value-k for name,value in margins.items()})
    assert len(slacks)==47 and len(margins)==7
    assert all(value>0 for value in slacks.values()),'non-positive exact assembly margin'
    assert {name:F(value) for name,value in A['strict_constraints'].items()}==slacks
    assert {name:F(value) for name,value in A['margins'].items()}==margins
    assert F(par['a_bit'])==a and F(par['kappa'])==k and F(par['eta'])==eta
    assert F(A['absorption_gap'])==F(r['absorption_gap'])==minimum-k
    assert F(r['next_grid_rejected'])==k+F(1,10**15) and F(r['next_grid_rejected'])>=minimum
    threshold=coarse/(1+coarse-old);assert threshold==F(d['threshold']) and atom>threshold
    ceiling=threshold/(1+2*threshold);assert ceiling==F(d['fixed_coarse_eta_zero_supremum']) and k<ceiling
    assert k>F(d['secondary']['kappa'])>F(d['headline']['kappa'])
    assert 0<eta<F(1,2) and 0<a<b<F(1,32)
    assert k<minimum<a/(1+2*a)<F(1,34)
    assert F(d['refined_coarse_control']['coarse_gap'])>0
    return dict(kappa=str(k),strict_constraints=47,positive_margins=7,assembly_family_bound='1/34',adapter_gap=str(atom-a),absorption_gap=str(minimum-k),
        improvement_over_headline_percent=float((k/F(d['headline']['kappa'])-1)*100),
        improvement_over_secondary_percent=float((k/F(d['secondary']['kappa'])-1)*100))
def main():
    d=json.loads((HERE/'candidate.json').read_text());result=validate(d);controls=[]
    for name,key,value in [('next-kappa','kappa',d['best']['record']['next_grid_rejected']),('zero-eta','eta','0'),('atom-equality','atom',d['threshold'])]:
        bad=copy.deepcopy(d);bad['best']['record'][key]=value
        try:validate(bad)
        except AssertionError:controls.append(name)
        else:raise AssertionError('corruption accepted: '+name)
    result['corruptions_rejected']=controls
    (HERE/'independent-check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
