/*
 * cg.h
 *
 *  Created on: Feb 22, 2013
 *      Author: Joao Cunha <joao.cunha@ua.pt>
 */

#ifndef CG_H_
#define CG_H_

#include "gp.h"

namespace libgp
{

class CG
{
public:
	CG();
	virtual ~CG();
	void maximize(GaussianProcess* gp, size_t n=100, bool verbose=1);
	void getLMLandDuration(double& lml, double& duration) { lml = lml_; duration = duration_; }
	void set_tolerance(double tol) { tol_ = tol; }
	void get_lml_time_history(std::vector<double>& lml_history, std::vector<double>& time_cost_history) {
		lml_history = lml_history_;
		time_cost_history = time_cost_history_;
	}
private:
	double lml_;
	double duration_;
	double tol_; // 新增：收敛容差成员变量
	std::vector<double> lml_history_;
	std::vector<double> time_cost_history_;
};

}

#endif /* CG_H_ */
